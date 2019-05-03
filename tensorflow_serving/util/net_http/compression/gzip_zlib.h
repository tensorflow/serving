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

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_COMPRESSION_GZIP_ZLIB_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_COMPRESSION_GZIP_ZLIB_H_

#include <zlib.h>

#include <cstdint>

namespace tensorflow {
namespace serving {
namespace net_http {

class GZipHeader {
 public:
  GZipHeader() { Reset(); }
  ~GZipHeader() {}

  // Wipe the slate clean and start from scratch.
  void Reset() {
    state_ = IN_HEADER_ID1;
    flags_ = 0;
    extra_length_ = 0;
  }

  enum Status {
    INCOMPLETE_HEADER,
    COMPLETE_HEADER,
    INVALID_HEADER,
  };

  // If the bytes we've seen so far do not yet constitute a complete gzip
  // header, return INCOMPLETE_HEADER. If these bytes do not constitute a valid
  // gzip header, return INVALID_HEADER. When we've seen a complete
  // gzip header, return COMPLETE_HEADER and set the pointer pointed
  // to by header_end to the first byte beyond the gzip header.
  Status ReadMore(const char *inbuf, int inbuf_len, const char **header_end);

 private:
  enum {                   // flags (see RFC)
    FLAG_FTEXT = 0x01,     // bit 0 set: file probably ascii text
    FLAG_FHCRC = 0x02,     // bit 1 set: header CRC present
    FLAG_FEXTRA = 0x04,    // bit 2 set: extra field present
    FLAG_FNAME = 0x08,     // bit 3 set: original file name present
    FLAG_FCOMMENT = 0x10,  // bit 4 set: file comment present
    FLAG_RESERVED = 0xE0,  // bits 5..7: reserved
  };

  enum State {
    // The first 10 bytes are the fixed-size header:
    IN_HEADER_ID1,
    IN_HEADER_ID2,
    IN_HEADER_CM,
    IN_HEADER_FLG,
    IN_HEADER_MTIME_BYTE_0,
    IN_HEADER_MTIME_BYTE_1,
    IN_HEADER_MTIME_BYTE_2,
    IN_HEADER_MTIME_BYTE_3,
    IN_HEADER_XFL,
    IN_HEADER_OS,

    IN_XLEN_BYTE_0,
    IN_XLEN_BYTE_1,
    IN_FEXTRA,

    IN_FNAME,

    IN_FCOMMENT,

    IN_FHCRC_BYTE_0,
    IN_FHCRC_BYTE_1,

    IN_DONE,
  };

  int state_;      // our current State in the parsing FSM: an int so we can ++
  uint8_t flags_;  // the flags byte of the header ("FLG" in the RFC)
  uint16_t extra_length_;  // how much of the "extra field" we have yet to read
};

class ZLib {
 public:
  ZLib();
  ~ZLib();

  // The max length of the buffer to store uncompressed data
  static constexpr int64_t kMaxUncompressedBytes = 100 * 1024 * 1024;  // 100MB

  // Wipe a ZLib object to a virgin state.  This differs from Reset()
  // in that it also breaks any dictionary, gzip, etc, state.
  void Reinit();

  // Call this to make a zlib buffer as good as new.  Here's the only
  // case where they differ:
  //    CompressChunk(a); CompressChunk(b); CompressChunkDone();   vs
  //    CompressChunk(a); Reset(); CompressChunk(b); CompressChunkDone();
  // You'll want to use Reset(), then, when you interrupt a compress
  // (or uncompress) in the middle of a chunk and want to start over.
  void Reset();

  // By default UncompressAtMostOrAll will return Z_OK upon hitting the end of
  // the input stream. This function modifies that behavior by returning
  // Z_STREAM_END instead. This is useful when getting multiple compressed
  // documents in a single stream. Returning Z_STREAM_END will indicate the end
  // of a document.
  void SetDontHideStreamEnd();

  // Sets the compression level to be used
  void SetCompressionLevel(int level) { settings_.compression_level_ = level; }

  // Sets the size of the window (history buffer) used by the compressor.
  // The size is expressed in bits (log base 2 of the desired size).
  void SetCompressionWindowSizeInBits(int bits) {
    settings_.window_bits_ = bits;
  }

  // Controls the amount of memory used by the compresser.
  // Legal value are 1 through 9. See zlib.h for more info.
  void SetCompressionMemLevel(int level) { settings_.mem_level_ = level; }

  // According to the zlib manual, when you Compress, the destination
  // buffer must have size at least src + .1%*src + 12.  This function
  // helps you calculate that.  Augment this to account for a potential
  // gzip header and footer, plus a few bytes of slack.
  static uLong MinCompressbufSize(uLong uncompress_size) {
    return uncompress_size + uncompress_size / 1000 + 40;
  }

  // The minimum size of footers written by CompressChunkDone().
  int MinFooterSize() const;

  // Compresses the source buffer into the destination buffer.
  // sourceLen is the byte length of the source buffer.
  // Upon entry, destLen is the total size of the destination buffer,
  // which must be of size at least MinCompressbufSize(sourceLen).
  // Upon exit, destLen is the actual size of the compressed buffer.
  //
  // This function can be used to compress a whole file at once if the
  // input file is mmap'ed.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer. Note that if the output buffer is exactly the same
  // size as the compressed result, we still return Z_BUF_ERROR.
  // (check CL#1936076)
  //
  // If the values of *destLen or sourceLen do not fit in an unsigned int,
  // Z_BUF_ERROR is returned.
  int Compress(Bytef *dest, uLongf *destLen, const Bytef *source,
               uLong sourceLen);

  // Uncompresses the source buffer into the destination buffer.
  // The destination buffer must be long enough to hold the entire
  // decompressed contents.
  //
  // Returns Z_OK on success, otherwise, it returns a zlib error code.
  //
  // If the values of *destLen or sourceLen do not fit in an unsigned int,
  // Z_BUF_ERROR is returned.
  int Uncompress(Bytef *dest, uLongf *destLen, const Bytef *source,
                 uLong sourceLen);

  // Get the uncompressed size from the gzip footer.
  uLongf GzipUncompressedLength(const Bytef *source, uLong len);

  // Special helper function to help uncompress gzipped documents:
  // We'll allocate (via std::allocator) a destination buffer exactly big
  // enough to hold the gzipped content.  We set dest and destLen.
  // If we don't return Z_OK, *dest will be NULL, otherwise you
  // should free() it when you're done with it.
  // Returns Z_OK on success, otherwise, it returns a zlib error code.
  // Its the responsibility of the user to set *destLen to the
  // expected maximum size of the uncompressed data. The size of the
  // uncompressed data is read from the compressed buffer gzip footer.
  // This value cannot be trusted, so we compare it to the expected
  // maximum size supplied by the user, returning Z_MEM_ERROR if its
  // greater than the expected maximum size.
  int UncompressGzipAndAllocate(Bytef **dest, uLongf *destLen,
                                const Bytef *source, uLong sourceLen);

  // Streaming compression and decompression methods.
  // {Unc,C}ompressAtMost() decrements sourceLen by the amount of data that was
  // consumed: if it returns Z_BUF_ERROR, set the source of the next
  // {Unc,C}ompressAtMost() to the unconsumed data.

  // Compresses data one chunk at a time -- ie you can call this more
  // than once.  This is useful for a webserver, for instance, which
  // might want to use chunked encoding with compression.  To get this
  // to work you need to call start and finish routines.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer.

  int CompressAtMost(Bytef *dest, uLongf *destLen, const Bytef *source,
                     uLong *sourceLen);

  // Emits gzip footer information, as needed.
  // destLen should be at least MinFooterSize() long.
  // Returns Z_OK, Z_MEM_ERROR, and Z_BUF_ERROR as in CompressChunk().
  int CompressChunkDone(Bytef *dest, uLongf *destLen);

  // Uncompress data one chunk at a time -- ie you can call this
  // more than once.  To get this to work you need to call per-chunk
  // and "done" routines.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer.

  int UncompressAtMost(Bytef *dest, uLongf *destLen, const Bytef *source,
                       uLong *sourceLen);

  // Checks gzip footer information, as needed.  Mostly this just
  // makes sure the checksums match.  Whenever you call this, it
  // will assume the last 8 bytes from the previous UncompressChunk
  // call are the footer.  Returns true iff everything looks ok.
  bool UncompressChunkDone();

  // Only meaningful for chunked compressing/uncompressing. It's true
  // after initialization or reset and before the first chunk of
  // user data is received.
  bool first_chunk() const { return first_chunk_; }

  // Convenience method to check if a bytestream has a header.  This
  // is intended as a quick test: "Is this likely a GZip file?"
  static bool HasGzipHeader(const char *source, int sourceLen);

  // Have we parsed the complete gzip footer? When this result is true, it is
  // time to call IsGzipFooterValid() / UncompressChunkDone().
  bool IsGzipFooterComplete() const;

  // Have we parsed the complete gzip footer, and does it match the
  // length and CRC checksum of the content that we have uncompressed
  // so far?
  bool IsGzipFooterValid() const;

  // Accessor for the uncompressed size
  uLong uncompressed_size() const { return uncompressed_size_; }

 private:
  int InflateInit();  // sets up the zlib inflate structure
  int DeflateInit();  // sets up the zlib deflate structure

  // These init the zlib data structures for compressing/uncompressing
  int CompressInit(Bytef *dest, uLongf *destLen, const Bytef *source,
                   uLong *sourceLen);
  int UncompressInit(Bytef *dest, uLongf *destLen, const Bytef *source,
                     uLong *sourceLen);
  // Initialization method to be called if we hit an error while
  // uncompressing. On hitting an error, call this method before
  // returning the error.
  void UncompressErrorInit();
  // Helper functions to write gzip-specific data
  int WriteGzipHeader();
  int WriteGzipFooter(Bytef *dest, uLongf destLen);

  // Helper function for both Compress and CompressChunk
  int CompressChunkOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                         uLong sourceLen, int flush_mode);
  int CompressAtMostOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                          uLong *sourceLen, int flush_mode);

  // Likewise for UncompressAndUncompressChunk
  int UncompressChunkOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                           uLong sourceLen, int flush_mode);

  int UncompressAtMostOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                            uLong *sourceLen, int flush_mode);

  // Initialization method to be called if we hit an error while
  // compressing. On hitting an error, call this method before
  // returning the error.
  void CompressErrorInit();

  struct Settings {
    // compression level
    int compression_level_;

    // log base 2 of the window size used in compression
    int window_bits_;

    // specifies the amount of memory to be used by compressor (1-9)
    int mem_level_;

    // Controls behavior of UncompressAtMostOrAll with regards to returning
    // Z_STREAM_END. See comments for SetDontHideStreamEnd.
    bool dont_hide_zstream_end_;
  };

  // "Current" settings. These will be used whenever we next configure zlib.
  // For example changing compression level or header mode will be recorded
  // in these, but don't usually get applied immediately but on next compress.
  Settings settings_;

  // Settings last used to initialise and configure zlib. These are needed
  // to know if the current desired configuration in settings_ is sufficiently
  // compatible with the previous configuration and we can just reconfigure the
  // underlying zlib objects, or have to recreate them from scratch.
  Settings init_settings_;

  z_stream comp_stream_;    // Zlib stream data structure
  bool comp_init_;          // True if we have initialized comp_stream_
  z_stream uncomp_stream_;  // Zlib stream data structure
  bool uncomp_init_;        // True if we have initialized uncomp_stream_

  // These are used only in gzip compression mode
  uLong crc_;  // stored in gzip footer, fitting 4 bytes
  uLong uncompressed_size_;

  GZipHeader *gzip_header_;  // our gzip header state

  Byte gzip_footer_[8];    // stored footer, used to uncompress
  int gzip_footer_bytes_;  // num of footer bytes read so far, or -1

  // These are used only with chunked compression.
  bool first_chunk_;  // true if we need to emit headers with this chunk
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_COMPRESSION_GZIP_ZLIB_H_
