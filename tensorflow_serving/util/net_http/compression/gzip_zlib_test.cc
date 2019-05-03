/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/util/net_http/compression/gzip_zlib.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace serving {
namespace net_http {
namespace {

std::random_device rd;

typedef std::mt19937_64 RandomEngine;

int GetUniformRand(RandomEngine* rng, int max) {
  std::uniform_int_distribution<int> uniform(0, max);
  return uniform(*rng);
}

// Take some test headers and pass them to a GZipHeader, fragmenting
// the headers in many different random ways.
TEST(GzipHeader, FragmentTest) {
  RandomEngine rng(rd());

  struct TestCase {
    const char* str;
    int len;        // total length of the string
    int cruft_len;  // length of the gzip header part
  };
  TestCase tests[] = {
      // Basic header:
      {"\037\213\010\000\216\176\356\075\002\003", 10, 0},

      // Basic headers with crud on the end:
      {"\037\213\010\000\216\176\356\075\002\003X", 11, 1},
      {"\037\213\010\000\216\176\356\075\002\003XXX", 13, 3},

      {
          "\037\213\010\010\321\135\265\100\000\003"
          "emacs\000",
          16, 0  // with an FNAME of "emacs"
      },
      {
          "\037\213\010\010\321\135\265\100\000\003"
          "\000",
          11, 0  // with an FNAME of zero bytes
      },
      {
          "\037\213\010\020\321\135\265\100\000\003"
          "emacs\000",
          16, 0,  // with an FCOMMENT of "emacs"
      },
      {
          "\037\213\010\020\321\135\265\100\000\003"
          "\000",
          11, 0,  // with an FCOMMENT of zero bytes
      },
      {
          "\037\213\010\002\321\135\265\100\000\003"
          "\001\002",
          12, 0  // with an FHCRC
      },
      {
          "\037\213\010\004\321\135\265\100\000\003"
          "\003\000foo",
          15, 0  // with an extra of "foo"
      },
      {
          "\037\213\010\004\321\135\265\100\000\003"
          "\000\000",
          12, 0  // with an extra of zero bytes
      },
      {
          "\037\213\010\032\321\135\265\100\000\003"
          "emacs\000"
          "emacs\000"
          "\001\002",
          24, 0  // with an FNAME of "emacs", FCOMMENT of "emacs", and FHCRC
      },
      {
          "\037\213\010\036\321\135\265\100\000\003"
          "\003\000foo"
          "emacs\000"
          "emacs\000"
          "\001\002",
          29, 0  // with an FNAME of "emacs", FCOMMENT of "emacs", FHCRC, "foo"
      },
      {
          "\037\213\010\036\321\135\265\100\000\003"
          "\003\000foo"
          "emacs\000"
          "emacs\000"
          "\001\002"
          "XXX",
          32, 3  // FNAME of "emacs", FCOMMENT of "emacs", FHCRC, "foo", crud
      },
  };

  // Test all the headers test cases.
  for (auto test : tests) {
    // Test many random ways they might be fragmented.
    for (int j = 0; j < 1000; ++j) {
      // Get the test case set up.
      const char* p = test.str;
      int bytes_left = test.len;
      int bytes_read = 0;

      // Pick some random places to fragment the headers.
      const int num_fragments = GetUniformRand(&rng, bytes_left);
      std::vector<int> fragment_starts;
      for (int frag_num = 0; frag_num < num_fragments; ++frag_num) {
        fragment_starts.push_back(GetUniformRand(&rng, bytes_left));
      }
      sort(fragment_starts.begin(), fragment_starts.end());

      GZipHeader gzip_headers;
      // Go through several fragments and pass them to the headers for parsing.
      int frag_num = 0;
      while (bytes_left > 0) {
        const int fragment_len = (frag_num < num_fragments)
                                     ? (fragment_starts[frag_num] - bytes_read)
                                     : (test.len - bytes_read);
        EXPECT_GE(fragment_len, 0);
        const char* header_end = nullptr;
        GZipHeader::Status status =
            gzip_headers.ReadMore(p, fragment_len, &header_end);
        bytes_read += fragment_len;
        bytes_left -= fragment_len;
        EXPECT_GE(bytes_left, 0);
        p += fragment_len;
        frag_num++;
        if (bytes_left <= test.cruft_len) {
          EXPECT_EQ(status, GZipHeader::COMPLETE_HEADER);
          break;
        } else {
          EXPECT_EQ(status, GZipHeader::INCOMPLETE_HEADER);
        }
      }  // while
    }    // for many fragmentations
  }      // for all test case headers
}

// 1048576 == 2^20 == 1 MB
#define MAX_BUF_SIZE 1048500
#define MAX_BUF_FLEX 1048576

void TestCompression(ZLib* zlib, const std::string& uncompbuf,
                     const char* msg) {
  uLongf complen = ZLib::MinCompressbufSize(uncompbuf.size());
  std::string compbuf(complen, '\0');
  int err = zlib->Compress((Bytef*)compbuf.data(), &complen,
                           (Bytef*)uncompbuf.data(), uncompbuf.size());
  EXPECT_EQ(Z_OK, err) << "  " << uncompbuf.size() << " bytes down to "
                       << complen << " bytes.";

  // Output data size should match input data size.
  uLongf uncomplen2 = uncompbuf.size();
  std::string uncompbuf2(uncomplen2, '\0');
  err = zlib->Uncompress((Bytef*)&uncompbuf2[0], &uncomplen2,
                         (Bytef*)compbuf.data(), complen);
  EXPECT_EQ(Z_OK, err);

  if (msg != nullptr) {
    printf("Orig: %7lu  Compressed: %7lu  %5.3f %s\n", uncomplen2, complen,
           (float)complen / uncomplen2, msg);
  }

  EXPECT_EQ(uncompbuf, absl::string_view(uncompbuf2.data(), uncomplen2))
      << "Uncompression mismatch!";
}

// Take some test inputs and pass them to zlib, fragmenting the input randomly.
void TestRandomGzipHeaderUncompress(ZLib* zlib) {
  RandomEngine rng(rd());

  struct TestCase {
    const char* str;
    int len;  // total length of the string
  };
  TestCase tests[] = {
      {
          // header, body ("hello, world!\n"), footer
          "\037\213\010\000\216\176\356\075\002\003"
          "\313\110\315\311\311\327\121\050\317\057\312\111\121\344\002\000"
          "\300\337\061\266\016\000\000\000",
          34,
      },
  };

  std::string uncompbuf2(MAX_BUF_FLEX, '\0');
  // Test all the headers test cases.
  for (uint32_t i = 0; i < ABSL_ARRAYSIZE(tests); ++i) {
    // Test many random ways they might be fragmented.
    for (int j = 0; j < 5 * 1000; ++j) {
      // Get the test case set up.
      const char* p = tests[i].str;
      int bytes_left = tests[i].len;
      int bytes_read = 0;
      int bytes_uncompressed = 0;
      zlib->Reset();

      // Pick some random places to fragment the headers.
      const int num_fragments = GetUniformRand(&rng, bytes_left);
      std::vector<int> fragment_starts;
      for (int frag_num = 0; frag_num < num_fragments; ++frag_num) {
        fragment_starts.push_back(GetUniformRand(&rng, bytes_left));
      }
      sort(fragment_starts.begin(), fragment_starts.end());

      // Go through several fragments and pass them in for parsing.
      int frag_num = 0;
      while (bytes_left > 0) {
        const int fragment_len = (frag_num < num_fragments)
                                     ? (fragment_starts[frag_num] - bytes_read)
                                     : (tests[i].len - bytes_read);
        ASSERT_GE(fragment_len, 0);
        if (fragment_len != 0) {  // zlib doesn't like 0-length buffers
          uLongf uncomplen2 = uncompbuf2.size() - bytes_uncompressed;
          auto complen_src = static_cast<uLongf>(fragment_len);
          int err = zlib->UncompressAtMost(
              (Bytef*)&uncompbuf2[0] + bytes_uncompressed, &uncomplen2,
              (const Bytef*)p, &complen_src);
          ASSERT_EQ(err, Z_OK);
          bytes_uncompressed += uncomplen2;
          bytes_read += fragment_len;
          bytes_left -= fragment_len;
          ASSERT_GE(bytes_left, 0);
          p += fragment_len;
        }
        frag_num++;
      }  // while bytes left to uncompress

      ASSERT_TRUE(zlib->UncompressChunkDone());
      EXPECT_EQ(sizeof("hello, world!\n") - 1, bytes_uncompressed);
      EXPECT_EQ(
          0, strncmp(uncompbuf2.data(), "hello, world!\n", bytes_uncompressed))
          << "Uncompression mismatch, expected 'hello, world!\\n', "
          << "got '" << absl::string_view(uncompbuf2.data(), bytes_uncompressed)
          << "'";
    }  // for many fragmentations
  }    // for all test case headers
}

constexpr int32_t kMaxSizeUncompressedData = 10 * 1024 * 1024;  // 10MB

void TestErrors(ZLib* zlib, const std::string& uncompbuf_str) {
  const char* uncompbuf = uncompbuf_str.data();
  const uLongf uncomplen = uncompbuf_str.size();
  std::string compbuf(MAX_BUF_SIZE, '\0');
  std::string uncompbuf2(MAX_BUF_FLEX, '\0');
  int err;

  uLongf complen = 23;  // don't give it enough space to compress
  err = zlib->Compress((Bytef*)compbuf.data(), &complen, (Bytef*)uncompbuf,
                       uncomplen);
  EXPECT_EQ(Z_BUF_ERROR, err);

  // OK, now successfully compress
  complen = compbuf.size();
  err = zlib->Compress((Bytef*)compbuf.data(), &complen, (Bytef*)uncompbuf,
                       uncomplen);
  EXPECT_EQ(Z_OK, err) << "  " << uncomplen << " bytes down to " << complen
                       << " bytes.";

  uLongf uncomplen2 = 10;  // not enough space to uncompress
  err = zlib->Uncompress((Bytef*)&uncompbuf2[0], &uncomplen2,
                         (Bytef*)compbuf.data(), complen);
  EXPECT_EQ(Z_BUF_ERROR, err);

  // Here we check what happens when we don't try to uncompress enough bytes
  uncomplen2 = uncompbuf2.size();
  err = zlib->Uncompress((Bytef*)&uncompbuf2[0], &uncomplen2,
                         (Bytef*)compbuf.data(), 23);
  EXPECT_EQ(Z_BUF_ERROR, err);

  uncomplen2 = uncompbuf2.size();
  uLongf comlen2 = 23;
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0], &uncomplen2,
                               (Bytef*)compbuf.data(), &comlen2);
  EXPECT_EQ(Z_OK, err);  // it's ok if a single chunk is too small
  if (err == Z_OK) {
    EXPECT_FALSE(zlib->UncompressChunkDone())
        << "UncompresDone() was happy with its 3 bytes of compressed data";
  }

  const int changepos = 0;
  const char oldval = compbuf[changepos];  // corrupt the input
  compbuf[changepos]++;
  uncomplen2 = uncompbuf2.size();
  err = zlib->Uncompress((Bytef*)&uncompbuf2[0], &uncomplen2,
                         (Bytef*)compbuf.data(), complen);
  EXPECT_NE(Z_OK, err);

  compbuf[changepos] = oldval;

  // Make sure our memory-allocating uncompressor deals with problems gracefully
  char* tmpbuf;
  char tmp_compbuf[10] = "\255\255\255\255\255\255\255\255\255";
  uncomplen2 = kMaxSizeUncompressedData;
  err = zlib->UncompressGzipAndAllocate(
      (Bytef**)&tmpbuf, &uncomplen2, (Bytef*)tmp_compbuf, sizeof(tmp_compbuf));
  EXPECT_NE(Z_OK, err);
  EXPECT_EQ(nullptr, tmpbuf);
}

void TestBogusGunzipRequest(ZLib* zlib) {
  const Bytef compbuf[] = "This is not compressed";
  const uLongf complen = sizeof(compbuf);
  Bytef* uncompbuf;
  uLongf uncomplen = 0;
  int err =
      zlib->UncompressGzipAndAllocate(&uncompbuf, &uncomplen, compbuf, complen);
  EXPECT_EQ(Z_DATA_ERROR, err);
}

void TestGzip(ZLib* zlib, const std::string& uncompbuf_str) {
  const char* uncompbuf = uncompbuf_str.data();
  const uLongf uncomplen = uncompbuf_str.size();
  std::string compbuf(MAX_BUF_SIZE, '\0');
  std::string uncompbuf2(MAX_BUF_FLEX, '\0');

  uLongf complen = compbuf.size();
  int err = zlib->Compress((Bytef*)compbuf.data(), &complen, (Bytef*)uncompbuf,
                           uncomplen);
  EXPECT_EQ(Z_OK, err) << "  " << uncomplen << " bytes down to " << complen
                       << " bytes.";

  uLongf uncomplen2 = uncompbuf2.size();
  err = zlib->Uncompress((Bytef*)&uncompbuf2[0], &uncomplen2,
                         (Bytef*)compbuf.data(), complen);
  EXPECT_EQ(Z_OK, err);
  EXPECT_EQ(uncomplen, uncomplen2) << "Uncompression mismatch!";
  EXPECT_EQ(0, memcmp(uncompbuf, uncompbuf2.data(), uncomplen))
      << "Uncompression mismatch!";

  // Also try the auto-allocate uncompressor
  char* tmpbuf;
  err = zlib->UncompressGzipAndAllocate((Bytef**)&tmpbuf, &uncomplen2,
                                        (Bytef*)compbuf.data(), complen);
  EXPECT_EQ(Z_OK, err);
  EXPECT_EQ(uncomplen, uncomplen2) << "Uncompression mismatch!";
  EXPECT_EQ(0, memcmp(uncompbuf, uncompbuf2.data(), uncomplen))
      << "Uncompression mismatch!";
  if (tmpbuf) {
    std::allocator<char>().deallocate(tmpbuf, uncomplen2);
  }
}

void TestChunkedGzip(ZLib* zlib, const std::string& uncompbuf_str,
                     int num_chunks) {
  const char* uncompbuf = uncompbuf_str.data();
  const uLongf uncomplen = uncompbuf_str.size();
  std::string compbuf(MAX_BUF_SIZE, '\0');
  std::string uncompbuf2(MAX_BUF_FLEX, '\0');
  EXPECT_GT(num_chunks, 2);

  // uncompbuf2 is larger than uncompbuf to test for decoding too much
  //
  // Note that it is possible to receive num_chunks+1 total
  // chunks, due to rounding error.
  const int chunklen = uncomplen / num_chunks;
  int chunknum, i, err;
  int cum_len[100];  // cumulative compressed length, max to 100
  cum_len[0] = 0;
  for (chunknum = 0, i = 0; i < uncomplen; i += chunklen, chunknum++) {
    uLongf complen = compbuf.size() - cum_len[chunknum];
    // Make sure the last chunk gets the correct chunksize.
    uLongf chunksize = (uncomplen - i) < chunklen ? (uncomplen - i) : chunklen;
    err = zlib->CompressAtMost((Bytef*)compbuf.data() + cum_len[chunknum],
                               &complen, (Bytef*)uncompbuf + i, &chunksize);
    ASSERT_EQ(Z_OK, err) << "  " << uncomplen << " bytes down to " << complen
                         << " bytes.";
    cum_len[chunknum + 1] = cum_len[chunknum] + complen;
  }
  uLongf complen = compbuf.size() - cum_len[chunknum];
  err = zlib->CompressChunkDone((Bytef*)compbuf.data() + cum_len[chunknum],
                                &complen);
  EXPECT_EQ(Z_OK, err);
  cum_len[chunknum + 1] = cum_len[chunknum] + complen;

  for (chunknum = 0, i = 0; i < uncomplen; i += chunklen, chunknum++) {
    uLongf uncomplen2 = uncomplen - i;
    // Make sure the last chunk gets the correct chunksize.
    int expected = uncomplen2 < chunklen ? uncomplen2 : chunklen;
    uLongf complen_src = cum_len[chunknum + 1] - cum_len[chunknum];
    err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0] + i, &uncomplen2,
                                 (Bytef*)compbuf.data() + cum_len[chunknum],
                                 &complen_src);
    EXPECT_EQ(Z_OK, err);
    EXPECT_EQ(expected, uncomplen2)
        << "Uncompress size is " << uncomplen2 << ", not " << expected;
  }
  // There should be no further uncompressed bytes, after uncomplen bytes.
  uLongf uncomplen2 = uncompbuf2.size() - uncomplen;
  EXPECT_NE(0, uncomplen2);
  uLongf complen_src = cum_len[chunknum + 1] - cum_len[chunknum];
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0] + uncomplen, &uncomplen2,
                               (Bytef*)compbuf.data() + cum_len[chunknum],
                               &complen_src);
  EXPECT_EQ(Z_OK, err);
  EXPECT_EQ(0, uncomplen2);
  EXPECT_TRUE(zlib->UncompressChunkDone());

  // Those uncomplen bytes should match.
  EXPECT_EQ(0, memcmp(uncompbuf, uncompbuf2.data(), uncomplen))
      << "Uncompression mismatch!";

  // Now test to make sure resetting works properly
  // (1) First, uncompress the first chunk and make sure it's ok
  uncomplen2 = uncompbuf2.size();
  complen_src = cum_len[1];
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0], &uncomplen2,
                               (Bytef*)compbuf.data(), &complen_src);
  EXPECT_EQ(Z_OK, err);
  EXPECT_EQ(chunklen, uncomplen2) << "Uncompression mismatch!";
  // The first uncomplen2 bytes should match, where uncomplen2 is the number of
  // successfully uncompressed bytes by the most recent UncompressChunk call.
  // The remaining (uncomplen - uncomplen2) bytes would still match if the
  // uncompression guaranteed not to modify the buffer other than those first
  // uncomplen2 bytes, but there is no such guarantee.
  EXPECT_EQ(0, memcmp(uncompbuf, uncompbuf2.data(), uncomplen2))
      << "Uncompression mismatch!";

  // (2) Now, try the first chunk again and see that there's an error
  uncomplen2 = uncompbuf2.size();
  complen_src = cum_len[1];
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0], &uncomplen2,
                               (Bytef*)compbuf.data(), &complen_src);
  EXPECT_EQ(Z_DATA_ERROR, err);

  // (3) Now reset it and try again, and see that it's ok
  zlib->Reset();
  uncomplen2 = uncompbuf2.size();
  complen_src = cum_len[1];
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0], &uncomplen2,
                               (Bytef*)compbuf.data(), &complen_src);
  EXPECT_EQ(Z_OK, err);
  EXPECT_EQ(chunklen, uncomplen2) << "Uncompression mismatch!";
  EXPECT_EQ(0, memcmp(uncompbuf, uncompbuf2.data(), uncomplen2))
      << "Uncompression mismatch!";

  // (4) Make sure we can tackle output buffers that are too small
  // with the *AtMost() interfaces.
  uLong source_len = cum_len[2] - cum_len[1];
  EXPECT_GT(source_len, 1);
  // uncomplen2 = source_len/2;
  uncomplen2 = 2;  // fixed as we use fixed strings now
  err = zlib->UncompressAtMost((Bytef*)&uncompbuf2[0], &uncomplen2,
                               (Bytef*)(compbuf.data() + cum_len[1]),
                               &source_len);
  EXPECT_EQ(Z_BUF_ERROR, err);

  EXPECT_EQ(0, memcmp(uncompbuf + chunklen, uncompbuf2.data(), uncomplen2))
      << "Uncompression mismatch!";

  const int saveuncomplen2 = uncomplen2;
  uncomplen2 = uncompbuf2.size() - uncomplen2;
  // Uncompress the rest of the chunk.
  err = zlib->UncompressAtMost(
      (Bytef*)&uncompbuf2[0], &uncomplen2,
      (Bytef*)(compbuf.data() + cum_len[2] - source_len), &source_len);

  EXPECT_EQ(Z_OK, err);

  EXPECT_EQ(0, memcmp(uncompbuf + chunklen + saveuncomplen2, uncompbuf2.data(),
                      uncomplen2))
      << "Uncompression mismatch!";

  // (5) Finally, reset again
  zlib->Reset();
}

void TestFooterBufferTooSmall(ZLib* zlib) {
  uLongf footer_len = zlib->MinFooterSize() - 1;
  ASSERT_EQ(9, footer_len);
  Bytef footer_buffer[9];
  int err = zlib->CompressChunkDone(footer_buffer, &footer_len);
  ASSERT_EQ(Z_BUF_ERROR, err);
  ASSERT_EQ(0, footer_len);
}

TEST(ZLibTest, HugeCompression) {
  // Just big enough to trigger 32 bit overflow in MinCompressbufSize()
  // calculation.
  const uLong HUGE_DATA_SIZE = 0x81000000;

  // Construct an easily compressible huge buffer.
  std::string uncompbuf(HUGE_DATA_SIZE, 'A');

  ZLib zlib;
  zlib.SetCompressionLevel(1);  //  as fast as possible
  TestCompression(&zlib, uncompbuf, nullptr);
}

// TODO(wenboz): random size randm data
const char kText[] = "1234567890abcdefghijklmnopqrstuvwxyz";

TEST(ZLibTest, Compression) {
  const std::string uncompbuf = kText;
  ZLib zlib;
  zlib.SetCompressionLevel(6);

  TestCompression(&zlib, uncompbuf, "fixed size");
}

TEST(ZLibTest, OtherErrors) {
  const std::string uncompbuf = kText;
  ZLib zlib;

  TestErrors(&zlib, uncompbuf);

  TestBogusGunzipRequest(&zlib);
}

TEST(ZLibTest, UncompressChunkedHeaders) {
  // TestGzipHeaderUncompress(&zlib);

  ZLib zlib;
  TestRandomGzipHeaderUncompress(&zlib);
}

TEST(ZLibTest, GzipCompression) {
  const std::string uncompbuf = kText;
  ZLib zlib;

  TestGzip(&zlib, uncompbuf);

  // Try compressing again using the same ZLib
  TestGzip(&zlib, uncompbuf);
}

TEST(ZLibTest, ChunkedCompression) {
  const std::string uncompbuf = kText;
  ZLib zlib;

  TestChunkedGzip(&zlib, uncompbuf, 5);

  // Try compressing again using the same ZLib
  TestChunkedGzip(&zlib, uncompbuf, 6);

  // In theory we can mix and match the type of compression we do
  TestGzip(&zlib, uncompbuf);
  TestChunkedGzip(&zlib, uncompbuf, 8);

  // Test writing final chunk and footer into buffer that's too small.
  TestFooterBufferTooSmall(&zlib);

  TestGzip(&zlib, uncompbuf);
}

TEST(ZLibTest, BytewiseRead) {
  std::string text =
      "v nedrah tundry vydra v getrah tyrit v vedrah yadra kedra";
  size_t text_len = text.size();
  size_t archive_len = ZLib::MinCompressbufSize(text_len);
  std::string archive(archive_len, '\0');
  size_t decompressed_len = text_len + 1;
  std::string decompressed(decompressed_len, '\0');
  size_t decompressed_offset = 0;

  ZLib compressor;
  int rc = compressor.Compress((Bytef*)archive.data(), &archive_len,
                               (Bytef*)text.data(), text_len);
  ASSERT_EQ(rc, Z_OK);

  ZLib zlib;
  for (size_t i = 0; i < archive_len; ++i) {
    size_t source_len = 1;
    size_t dest_len = decompressed_len - decompressed_offset;
    rc = zlib.UncompressAtMost(
        (Bytef*)decompressed.data() + decompressed_offset, &dest_len,
        (Bytef*)archive.data() + i, &source_len);
    ASSERT_EQ(rc, Z_OK);
    ASSERT_EQ(source_len, 0);
    decompressed_offset += dest_len;
  }

  ASSERT_TRUE(zlib.IsGzipFooterValid());
  ASSERT_EQ(decompressed_offset, text_len);

  std::string truncated_output(decompressed.data(), text_len);
  ASSERT_EQ(truncated_output, text);
}

}  // namespace
}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
