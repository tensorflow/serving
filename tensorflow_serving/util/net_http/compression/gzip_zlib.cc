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

// zlib based C++ wrapper to support gzip compression/uncompression

#include "tensorflow_serving/util/net_http/compression/gzip_zlib.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>

#include "absl/base/casts.h"
#include "absl/base/macros.h"
#include "tensorflow_serving/util/net_http/internal/net_logging.h"

namespace tensorflow {
namespace serving {
namespace net_http {

// TODO(wenboz): disable setting change, free-list (no setting change)

// The GZIP header (see RFC 1952):
//   +---+---+---+---+---+---+---+---+---+---+
//   |ID1|ID2|CM |FLG|     MTIME     |XFL|OS |
//   +---+---+---+---+---+---+---+---+---+---+
//     ID1     \037
//     ID2     \213
//     CM      \010 (compression method == DEFLATE)
//     FLG     \000 (special flags that we do not support)
//     MTIME   Unix format modification time (0 means not available)
//     XFL     2-4? DEFLATE flags
//     OS      ???? Operating system indicator (255 means unknown)

constexpr char GZIP_HEADER[] = "\037\213\010\000\000\000\000\000\002\377";
constexpr uint8_t kMagicHeader[2] = {0x1f, 0x8b};  // gzip magic header

GZipHeader::Status GZipHeader::ReadMore(const char *inbuf, int inbuf_len,
                                        const char **header_end) {
  auto pos = reinterpret_cast<const uint8_t *>(inbuf);
  const uint8_t *const end = pos + inbuf_len;

  while (pos < end) {
    switch (state_) {
      case IN_HEADER_ID1:
        if (*pos != kMagicHeader[0]) return INVALID_HEADER;
        pos++;
        state_++;
        break;
      case IN_HEADER_ID2:
        if (*pos != kMagicHeader[1]) return INVALID_HEADER;
        pos++;
        state_++;
        break;
      case IN_HEADER_CM:
        if (*pos != Z_DEFLATED) return INVALID_HEADER;
        pos++;
        state_++;
        break;
      case IN_HEADER_FLG:
        flags_ =
            (*pos) & (FLAG_FHCRC | FLAG_FEXTRA | FLAG_FNAME | FLAG_FCOMMENT);
        pos++;
        state_++;
        break;

      case IN_HEADER_MTIME_BYTE_0:
        pos++;
        state_++;
        break;
      case IN_HEADER_MTIME_BYTE_1:
        pos++;
        state_++;
        break;
      case IN_HEADER_MTIME_BYTE_2:
        pos++;
        state_++;
        break;
      case IN_HEADER_MTIME_BYTE_3:
        pos++;
        state_++;
        break;

      case IN_HEADER_XFL:
        pos++;
        state_++;
        break;

      case IN_HEADER_OS:
        pos++;
        state_++;
        break;

      case IN_XLEN_BYTE_0:
        if (!(flags_ & FLAG_FEXTRA)) {
          state_ = IN_FNAME;
          break;
        }
        // We have a two-byte little-endian length, followed by a
        // field of that length.
        extra_length_ = *pos;
        pos++;
        state_++;
        break;
      case IN_XLEN_BYTE_1:
        extra_length_ += (*pos) << 8;
        pos++;
        state_++;
        ABSL_FALLTHROUGH_INTENDED;
        // if we have a zero-length FEXTRA, we want to check
        // to notice that we're done reading the FEXTRA before we exit the loop.

      case IN_FEXTRA: {
        // Grab the rest of the bytes in the extra field, or as many
        // of them as are actually present so far.
        const int num_extra_bytes =
            std::min<int>(extra_length_, absl::implicit_cast<int>(end - pos));
        pos += num_extra_bytes;
        extra_length_ -= num_extra_bytes;
        if (extra_length_ == 0) {
          state_ = IN_FNAME;  // advance when we've seen extra_length_ bytes
          flags_ &= ~FLAG_FEXTRA;  // we're done with the FEXTRA stuff
        }
        break;
      }

      case IN_FNAME:
        if (!(flags_ & FLAG_FNAME)) {
          state_ = IN_FCOMMENT;
          break;
        }
        // See if we can find the end of the \0-terminated FNAME field.
        pos = reinterpret_cast<const uint8_t *>(memchr(pos, '\0', (end - pos)));
        if (pos != nullptr) {
          pos++;                  // advance past the '\0'
          flags_ &= ~FLAG_FNAME;  // we're done with the FNAME stuff
          state_ = IN_FCOMMENT;
        } else {
          pos = end;  // everything we have so far is part of the FNAME
        }
        break;

      case IN_FCOMMENT:
        if (!(flags_ & FLAG_FCOMMENT)) {
          state_ = IN_FHCRC_BYTE_0;
          break;
        }
        // See if we can find the end of the \0-terminated FCOMMENT field.
        pos = reinterpret_cast<const uint8_t *>(memchr(pos, '\0', (end - pos)));
        if (pos != nullptr) {
          pos++;                     // advance past the '\0'
          flags_ &= ~FLAG_FCOMMENT;  // we're done with the FCOMMENT stuff
          state_ = IN_FHCRC_BYTE_0;
        } else {
          pos = end;  // everything we have so far is part of the FNAME
        }
        break;

      case IN_FHCRC_BYTE_0:
        if (!(flags_ & FLAG_FHCRC)) {
          state_ = IN_DONE;
          break;
        }
        pos++;
        state_++;
        break;

      case IN_FHCRC_BYTE_1:
        pos++;
        flags_ &= ~FLAG_FHCRC;  // we're done with the FHCRC stuff
        state_++;
        break;

      case IN_DONE:
        *header_end = reinterpret_cast<const char *>(pos);
        return COMPLETE_HEADER;

      default:
        break;
    }
  }

  if ((state_ > IN_HEADER_OS) && (flags_ == 0)) {
    *header_end = reinterpret_cast<const char *>(pos);
    return COMPLETE_HEADER;
  } else {
    return INCOMPLETE_HEADER;
  }
}

ZLib::ZLib()
    : comp_init_(false), uncomp_init_(false), gzip_header_(new GZipHeader) {
  Reinit();
  init_settings_ = settings_;
}

ZLib::~ZLib() {
  if (comp_init_) {
    deflateEnd(&comp_stream_);
  }
  if (uncomp_init_) {
    inflateEnd(&uncomp_stream_);
  }
  delete gzip_header_;
}

void ZLib::Reinit() {
  settings_.compression_level_ = Z_DEFAULT_COMPRESSION;
  settings_.window_bits_ = MAX_WBITS;
  settings_.mem_level_ = 8;  // DEF_MEM_LEVEL
  settings_.dont_hide_zstream_end_ = false;

  if (comp_init_) {
    int err = deflateReset(&comp_stream_);
    if (err != Z_OK) {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }
  if (uncomp_init_) {
    // Use negative window bits size to indicate bare stream with no header.
    int wbits = -MAX_WBITS;
    int err = inflateReset2(&uncomp_stream_, wbits);
    if (err != Z_OK) {
      inflateEnd(&uncomp_stream_);
      uncomp_init_ = false;
    }
  }
  crc_ = 0;
  uncompressed_size_ = 0;
  gzip_header_->Reset();
  gzip_footer_bytes_ = -1;
  first_chunk_ = true;
}

void ZLib::Reset() {
  first_chunk_ = true;
  gzip_header_->Reset();
}

void ZLib::SetDontHideStreamEnd() { settings_.dont_hide_zstream_end_ = true; }

int ZLib::MinFooterSize() const {
  int min_footer_size = 2;  // Room for empty chunk.
  min_footer_size += 8;     // Room for actual footer for gzip
  return min_footer_size;
}

// --------- COMPRESS MODE

// Initialization method to be called if we hit an error while
// compressing. On hitting an error, call this method before returning
// the error.
void ZLib::CompressErrorInit() {
  if (comp_init_) {
    deflateEnd(&comp_stream_);
    comp_init_ = false;
  }
  Reset();
}

// These probably return Z_OK, but may return Z_BUF_ERROR if outbuf is full
int ZLib::WriteGzipHeader() {
  if (comp_stream_.avail_out < sizeof(GZIP_HEADER)) return Z_BUF_ERROR;
  memcpy(comp_stream_.next_out, GZIP_HEADER, sizeof(GZIP_HEADER) - 1);
  comp_stream_.next_out += sizeof(GZIP_HEADER) - 1;
  comp_stream_.avail_out -= sizeof(GZIP_HEADER) - 1;
  return Z_OK;
}

int ZLib::WriteGzipFooter(Bytef *dest, uLongf destLen) {
  if (destLen < 8)  // not enough space for footer
    return Z_BUF_ERROR;
  *dest++ = (crc_ >> 0) & 255;
  *dest++ = (crc_ >> 8) & 255;
  *dest++ = (crc_ >> 16) & 255;
  *dest++ = (crc_ >> 24) & 255;
  *dest++ = (uncompressed_size_ >> 0) & 255;
  *dest++ = (uncompressed_size_ >> 8) & 255;
  *dest++ = (uncompressed_size_ >> 16) & 255;
  *dest++ = (uncompressed_size_ >> 24) & 255;
  return Z_OK;
}

int ZLib::DeflateInit() {
  int err = deflateInit2(&comp_stream_, settings_.compression_level_,
                         Z_DEFLATED, -settings_.window_bits_,
                         settings_.mem_level_, Z_DEFAULT_STRATEGY);
  if (err == Z_OK) {
    // Save parameters for later reusability checks
    init_settings_.compression_level_ = settings_.compression_level_;
    init_settings_.window_bits_ = settings_.window_bits_;
    init_settings_.mem_level_ = settings_.mem_level_;
  }
  return err;
}

int ZLib::CompressInit(Bytef *dest, uLongf *destLen, const Bytef *source,
                       uLong *sourceLen) {
  int err;

  comp_stream_.next_in = (Bytef *)source;
  comp_stream_.avail_in = (uInt)*sourceLen;
  // Check for sourceLen (unsigned long) to fit into avail_in (unsigned int).
  if ((uLong)comp_stream_.avail_in != *sourceLen) return Z_BUF_ERROR;
  comp_stream_.next_out = dest;
  comp_stream_.avail_out = (uInt)*destLen;
  // Check for destLen (unsigned long) to fit into avail_out (unsigned int).
  if ((uLong)comp_stream_.avail_out != *destLen) return Z_BUF_ERROR;

  if (!first_chunk_)  // only need to set up stream the first time through
    return Z_OK;

  // Force full reinit if properties have changed in a way we can't adjust.
  if (comp_init_ && (init_settings_.window_bits_ != settings_.window_bits_ ||
                     init_settings_.mem_level_ != settings_.mem_level_)) {
    deflateEnd(&comp_stream_);
    comp_init_ = false;
  }

  // Reuse if we've already initted the object.
  if (comp_init_) {  // we've already initted it
    err = deflateReset(&comp_stream_);
    if (err != Z_OK) {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }

  // If compression level has changed, try to reconfigure instead of reinit
  if (comp_init_ &&
      init_settings_.compression_level_ != settings_.compression_level_) {
    err = deflateParams(&comp_stream_, settings_.compression_level_,
                        Z_DEFAULT_STRATEGY);
    if (err == Z_OK) {
      init_settings_.compression_level_ = settings_.compression_level_;
    } else {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }

  // First use or previous state was not reusable with current settings.
  if (!comp_init_) {
    comp_stream_.zalloc = (alloc_func)0;
    comp_stream_.zfree = (free_func)0;
    comp_stream_.opaque = (voidpf)0;
    err = DeflateInit();
    if (err != Z_OK) return err;
    comp_init_ = true;
  }
  return Z_OK;
}

// Supports chunked compression, using the chunked compression features of zlib.
int ZLib::CompressAtMostOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                              uLong *sourceLen,
                              int flush_mode) {  // Z_FULL_FLUSH or Z_FINISH
  int err;

  if ((err = CompressInit(dest, destLen, source, sourceLen)) != Z_OK)
    return err;

  // This is used to figure out how many bytes we wrote *this chunk*
  uint64_t compressed_size = comp_stream_.total_out;

  // Some setup happens only for the first chunk we compress in a run
  if (first_chunk_) {
    if ((err = WriteGzipHeader()) != Z_OK) return err;
    compressed_size -= sizeof(GZIP_HEADER) - 1;  // -= is right: adds to size
    crc_ = crc32(0, nullptr, 0);                 // initialize

    uncompressed_size_ = 0;
    first_chunk_ = false;  // so we don't do this again
  }

  // flush_mode is Z_FINISH for all mode, Z_SYNC_FLUSH for incremental
  // compression.
  err = deflate(&comp_stream_, flush_mode);

  const uLong source_bytes_consumed = *sourceLen - comp_stream_.avail_in;
  *sourceLen = comp_stream_.avail_in;

  if ((err == Z_STREAM_END || err == Z_OK) && comp_stream_.avail_in == 0 &&
      comp_stream_.avail_out != 0) {
    // we processed everything ok and the output buffer was large enough.
  } else if (err == Z_STREAM_END && comp_stream_.avail_in > 0) {
    return Z_BUF_ERROR;  // should never happen
  } else if (err != Z_OK && err != Z_STREAM_END && err != Z_BUF_ERROR) {
    // an error happened
    CompressErrorInit();
    return err;
  } else if (comp_stream_.avail_out == 0) {  // not enough space
    err = Z_BUF_ERROR;
  }

  assert(err == Z_OK || err == Z_STREAM_END || err == Z_BUF_ERROR);
  if (err == Z_STREAM_END) err = Z_OK;

  // update the crc and other metadata
  uncompressed_size_ += source_bytes_consumed;
  compressed_size = comp_stream_.total_out - compressed_size;  // delta
  *destLen = compressed_size;

  crc_ = crc32(crc_, source, source_bytes_consumed);

  return err;
}

int ZLib::CompressChunkOrAll(Bytef *dest, uLongf *destLen, const Bytef *source,
                             uLong sourceLen,
                             int flush_mode) {  // Z_FULL_FLUSH or Z_FINISH
  const int ret =
      CompressAtMostOrAll(dest, destLen, source, &sourceLen, flush_mode);
  if (ret == Z_BUF_ERROR) CompressErrorInit();
  return ret;
}

int ZLib::CompressAtMost(Bytef *dest, uLongf *destLen, const Bytef *source,
                         uLong *sourceLen) {
  return CompressAtMostOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

// This writes the gzip footer info, if necessary.
// No matter what, we call Reset() so we can compress Chunks again.
int ZLib::CompressChunkDone(Bytef *dest, uLongf *destLen) {
  // Make sure our buffer is of reasonable size.
  if (*destLen < static_cast<uLongf>(MinFooterSize())) {
    *destLen = 0;
    return Z_BUF_ERROR;
  }

  // The underlying zlib library requires a non-NULL source pointer, even if the
  // source length is zero, otherwise it will generate an (incorrect) zero-
  // valued CRC checksum.
  char dummy = '\0';
  int err;

  assert(!first_chunk_ && comp_init_);

  const uLongf orig_destLen = *destLen;
  if ((err = CompressChunkOrAll(dest, destLen, (const Bytef *)&dummy, 0,
                                Z_FINISH)) != Z_OK) {
    Reset();  // we assume they won't retry on error
    return err;
  }

  // Make sure that when we exit, we can start a new round of chunks later
  // (This must be set after the call to CompressChunkOrAll() above.)
  Reset();

  // Write gzip footer.  They're explicitly in little-endian order
  if ((err = WriteGzipFooter(dest + *destLen, orig_destLen - *destLen)) != Z_OK)
    return err;
  *destLen += 8;  // zlib footer took up another 8 bytes

  return Z_OK;  // stream_end is ok
}

// This routine only initializes the compression stream once.  Thereafter, it
// just does a deflateReset on the stream, which should be faster.
int ZLib::Compress(Bytef *dest, uLongf *destLen, const Bytef *source,
                   uLong sourceLen) {
  int err;
  const uLongf orig_destLen = *destLen;
  if ((err = CompressChunkOrAll(dest, destLen, source, sourceLen, Z_FINISH)) !=
      Z_OK)
    return err;
  Reset();  // reset for next call to Compress

  if ((err = WriteGzipFooter(dest + *destLen, orig_destLen - *destLen)) != Z_OK)
    return err;
  *destLen += 8;  // zlib footer took up another 8 bytes

  return Z_OK;
}

// --------- UNCOMPRESS MODE

int ZLib::InflateInit() {
  // Use negative window bits size to indicate bare stream with no header.
  int wbits = (-MAX_WBITS);
  int err = inflateInit2(&uncomp_stream_, wbits);
  return err;
}

// Initialization method to be called if we hit an error while
// uncompressing. On hitting an error, call this method before
// returning the error.
void ZLib::UncompressErrorInit() {
  if (uncomp_init_) {
    inflateEnd(&uncomp_stream_);
    uncomp_init_ = false;
  }
  Reset();
}

int ZLib::UncompressInit(Bytef *dest, uLongf *destLen, const Bytef *source,
                         uLong *sourceLen) {
  int err;

  uncomp_stream_.next_in = (Bytef *)source;
  uncomp_stream_.avail_in = (uInt)*sourceLen;
  // Check for sourceLen (unsigned long) to fit into avail_in (unsigned int).
  if ((uLong)uncomp_stream_.avail_in != *sourceLen) return Z_BUF_ERROR;

  uncomp_stream_.next_out = dest;
  uncomp_stream_.avail_out = (uInt)*destLen;
  // Check for destLen (unsigned long) to fit into avail_out (unsigned int).
  if ((uLong)uncomp_stream_.avail_out != *destLen) return Z_BUF_ERROR;

  if (!first_chunk_)  // only need to set up stream the first time through
    return Z_OK;

  // Reuse if we've already initted the object.
  if (uncomp_init_) {
    // Use negative window bits size to indicate bare stream with no header.
    int wbits = -MAX_WBITS;
    err = inflateReset2(&uncomp_stream_, wbits);
    if (err != Z_OK) {
      UncompressErrorInit();
    }
  }

  // First use or previous state was not reusable with current settings.
  if (!uncomp_init_) {
    uncomp_stream_.zalloc = (alloc_func)0;
    uncomp_stream_.zfree = (free_func)0;
    uncomp_stream_.opaque = (voidpf)0;
    err = InflateInit();
    if (err != Z_OK) return err;
    uncomp_init_ = true;
  }
  return Z_OK;
}

// If you compressed your data a chunk at a time, with CompressChunk,
// you can uncompress it a chunk at a time with UncompressChunk.
// Only difference between chunked and unchunked uncompression
// is the flush mode we use: Z_SYNC_FLUSH (chunked) or Z_FINISH (unchunked).
int ZLib::UncompressAtMostOrAll(Bytef *dest, uLongf *destLen,
                                const Bytef *source, uLong *sourceLen,
                                int flush_mode) {  // Z_SYNC_FLUSH or Z_FINISH
  int err = Z_OK;

  if (first_chunk_) {
    gzip_footer_bytes_ = -1;

    // If we haven't read our first chunk of actual compressed data,
    // and we're expecting gzip headers, then parse some more bytes
    // from the gzip headers.
    const Bytef *bodyBegin = nullptr;
    GZipHeader::Status status = gzip_header_->ReadMore(
        reinterpret_cast<const char *>(source), *sourceLen,
        reinterpret_cast<const char **>(&bodyBegin));
    switch (status) {
      case GZipHeader::INCOMPLETE_HEADER:  // don't have the complete header
        *destLen = 0;
        *sourceLen = 0;  // GZipHeader used all the input
        return Z_OK;
      case GZipHeader::INVALID_HEADER:  // bogus header
        Reset();
        return Z_DATA_ERROR;
      case GZipHeader::COMPLETE_HEADER:      // we have the full header
        *sourceLen -= (bodyBegin - source);  // skip past header bytes
        source = bodyBegin;
        crc_ = crc32(0, nullptr, 0);  // initialize CRC
        break;
      default:
        NET_LOG(FATAL, "Unexpected gzip header parsing result: %d", status);
    }
  } else if (gzip_footer_bytes_ >= 0) {
    // We're now just reading the gzip footer. We already read all the data.
    if (gzip_footer_bytes_ + *sourceLen > sizeof(gzip_footer_)) {
      Reset();
      return Z_DATA_ERROR;
    }
    uLong len = sizeof(gzip_footer_) - gzip_footer_bytes_;
    if (len > *sourceLen) len = *sourceLen;
    if (len > 0) {
      memcpy(gzip_footer_ + gzip_footer_bytes_, source, len);
      gzip_footer_bytes_ += len;
    }
    *sourceLen -= len;
    *destLen = 0;
    return Z_OK;
  }

  if ((err = UncompressInit(dest, destLen, source, sourceLen)) != Z_OK) {
    NET_LOG(WARNING,
            "UncompressInit: Error: %d "
            " SourceLen: %zu",
            err, *sourceLen);
    return err;
  }

  // This is used to figure out how many output bytes we wrote *this chunk*:
  const uLong old_total_out = uncomp_stream_.total_out;

  // This is used to figure out how many input bytes we read *this chunk*:
  const uLong old_total_in = uncomp_stream_.total_in;

  if (first_chunk_) {
    first_chunk_ = false;  // so we don't do this again

    // For the first chunk *only* (to avoid infinite troubles), we let
    // there be no actual data to uncompress.  This sometimes triggers
    // when the input is only the gzip header.
    if (*sourceLen == 0) {
      *destLen = 0;
      return Z_OK;
    }
  }

  // We'll uncompress as much as we can.  If we end OK great, otherwise
  // if we get an error that seems to be the gzip footer, we store the
  // gzip footer and return OK, otherwise we return the error.

  // flush_mode is Z_SYNC_FLUSH for chunked mode, Z_FINISH for all mode.
  err = inflate(&uncomp_stream_, flush_mode);

  // Figure out how many bytes of the input zlib slurped up:
  const uLong bytes_read = uncomp_stream_.total_in - old_total_in;
  assert((source + bytes_read) <= (source + *sourceLen));
  *sourceLen = uncomp_stream_.avail_in;

  // Next we look at the footer, if any. Note that we might currently
  // have just part of the footer (eg, if this data is arriving over a
  // socket). After looking for a footer, log a warning if there is data.
  if ((err == Z_STREAM_END) &&
      ((gzip_footer_bytes_ == -1) ||
       (static_cast<size_t>(gzip_footer_bytes_) < sizeof(gzip_footer_))) &&
      (uncomp_stream_.avail_in <= sizeof(gzip_footer_))) {
    // Store gzip footer bytes so we can check for footer consistency
    // in UncompressChunkDone(). (If we have the whole footer, we
    // could do the checking here, but we don't to keep consistency
    // with CompressChunkDone().)
    gzip_footer_bytes_ =
        std::min(absl::implicit_cast<size_t>(uncomp_stream_.avail_in),
                 sizeof(gzip_footer_));
    memcpy(gzip_footer_, source + bytes_read, gzip_footer_bytes_);
    *sourceLen -= gzip_footer_bytes_;
  } else if ((err == Z_STREAM_END || err == Z_OK)  // everything went ok
             && uncomp_stream_.avail_in == 0) {    // and we read it all
  } else if (err == Z_STREAM_END && uncomp_stream_.avail_in > 0) {
    UncompressErrorInit();
    return Z_DATA_ERROR;
  } else if (err != Z_OK && err != Z_STREAM_END && err != Z_BUF_ERROR) {
    UncompressErrorInit();
    return err;
  } else if (uncomp_stream_.avail_out == 0) {
    err = Z_BUF_ERROR;
  }

  assert(err == Z_OK || err == Z_BUF_ERROR || err == Z_STREAM_END);
  if (err == Z_STREAM_END && !settings_.dont_hide_zstream_end_) err = Z_OK;

  // update the crc and other metadata
  uncompressed_size_ = uncomp_stream_.total_out;
  *destLen = uncomp_stream_.total_out - old_total_out;  // size for this call

  crc_ = crc32(crc_, dest, *destLen);

  return err;
}

int ZLib::UncompressChunkOrAll(Bytef *dest, uLongf *destLen,
                               const Bytef *source, uLong sourceLen,
                               int flush_mode) {  // Z_SYNC_FLUSH or Z_FINISH
  const int ret =
      UncompressAtMostOrAll(dest, destLen, source, &sourceLen, flush_mode);
  if (ret == Z_BUF_ERROR) UncompressErrorInit();
  return ret;
}

int ZLib::UncompressAtMost(Bytef *dest, uLongf *destLen, const Bytef *source,
                           uLong *sourceLen) {
  return UncompressAtMostOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

// We make sure we've uncompressed everything, that is, the current
// uncompress stream is at a compressed-buffer-EOF boundary.  In gzip
// mode, we also check the gzip footer to make sure we pass the gzip
// consistency checks.  We RETURN true iff both types of checks pass.
bool ZLib::UncompressChunkDone() {
  assert(!first_chunk_ && uncomp_init_);
  // Make sure we're at the end-of-compressed-data point.  This means
  // if we call inflate with Z_FINISH we won't consume any input or
  // write any output
  Bytef dummyin, dummyout;
  uLongf dummylen = 0;
  if (UncompressChunkOrAll(&dummyout, &dummylen, &dummyin, 0, Z_FINISH) !=
      Z_OK) {
    return false;
  }

  // Make sure that when we exit, we can start a new round of chunks later
  Reset();

  // Whether we were hoping for a gzip footer or not, we allow a gzip
  // footer.  (See the note above about bugs in old zlibwrappers.) But
  // by the time we've seen all the input, it has to be either a
  // complete gzip footer, or no footer at all.
  if ((gzip_footer_bytes_ != -1) && (gzip_footer_bytes_ != 0) &&
      (static_cast<size_t>(gzip_footer_bytes_) != sizeof(gzip_footer_)))
    return false;

  return IsGzipFooterValid();
}

bool ZLib::IsGzipFooterComplete() const {
  return gzip_footer_bytes_ != -1 &&
         static_cast<size_t>(gzip_footer_bytes_) >= sizeof(gzip_footer_);
}

bool ZLib::IsGzipFooterValid() const {
  if (!IsGzipFooterComplete()) return false;

  // The footer holds the lower four bytes of the length.
  uLong uncompressed_size = 0;
  uncompressed_size += static_cast<uLong>(gzip_footer_[7]) << 24;
  uncompressed_size += gzip_footer_[6] << 16;
  uncompressed_size += gzip_footer_[5] << 8;
  uncompressed_size += gzip_footer_[4] << 0;
  if (uncompressed_size != (uncompressed_size_ & 0xffffffff)) {
    return false;
  }

  uLong checksum = 0;
  checksum += static_cast<uLong>(gzip_footer_[3]) << 24;
  checksum += gzip_footer_[2] << 16;
  checksum += gzip_footer_[1] << 8;
  checksum += gzip_footer_[0] << 0;
  if (crc_ != checksum) return false;

  return true;
}

// Uncompresses the source buffer into the destination buffer.
// The destination buffer must be long enough to hold the entire
// decompressed contents.
//
// We only initialize the uncomp_stream once.  Thereafter, we use
// inflateReset2, which should be faster.
//
// Returns Z_OK on success, otherwise, it returns a zlib error code.
int ZLib::Uncompress(Bytef *dest, uLongf *destLen, const Bytef *source,
                     uLong sourceLen) {
  int err;
  if ((err = UncompressChunkOrAll(dest, destLen, source, sourceLen,
                                  Z_FINISH)) != Z_OK) {
    Reset();  // let us try to compress again
    return err;
  }
  if (!UncompressChunkDone())  // calls Reset()
    return Z_DATA_ERROR;
  return Z_OK;  // stream_end is ok
}

// read uncompress length from gzip footer
uLongf ZLib::GzipUncompressedLength(const Bytef *source, uLong len) {
  assert(len > 4);
  return (static_cast<uLongf>(source[len - 1]) << 24) +
         (static_cast<uLongf>(source[len - 2]) << 16) +
         (static_cast<uLongf>(source[len - 3]) << 8) +
         (static_cast<uLongf>(source[len - 4]) << 0);
}

int ZLib::UncompressGzipAndAllocate(Bytef **dest, uLongf *destLen,
                                    const Bytef *source, uLong sourceLen) {
  *dest = nullptr;  // until we successfully allocate

  uLongf uncompress_length = GzipUncompressedLength(source, sourceLen);

  // Do not trust the uncompress size reported by the compressed buffer.
  if (uncompress_length > *destLen) {
    if (!HasGzipHeader(reinterpret_cast<const char *>(source), sourceLen)) {
      return Z_DATA_ERROR;
    }
    return Z_MEM_ERROR;  // probably a corrupted gzip buffer
  }

  *destLen = uncompress_length;

  *dest = std::allocator<Bytef>().allocate(*destLen);
  if (*dest == nullptr) {
    return Z_MEM_ERROR;
  }

  const int retval = Uncompress(*dest, destLen, source, sourceLen);
  if (retval != Z_OK) {  // just to make life easier for them
    std::allocator<Bytef>().deallocate(*dest, *destLen);
    *dest = nullptr;
  }
  return retval;
}

// Convenience method to check if a bytestream has a gzip header.
bool ZLib::HasGzipHeader(const char *source, int sourceLen) {
  GZipHeader gzh;
  const char *ptr = nullptr;
  return gzh.ReadMore(source, sourceLen, &ptr) == GZipHeader::COMPLETE_HEADER;
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
