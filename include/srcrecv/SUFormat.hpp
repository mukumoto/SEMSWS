/**
 * @file SUFormat.hpp
 * @brief Seismic Unix (SU) trace header definition for receiver output
 *
 * Self-contained 240-byte SU trace header struct.
 * Does not depend on external SU/CWP headers.
 * All I/O uses native (little-endian) byte order.
 */

#ifndef SEM_SU_FORMAT_HPP
#define SEM_SU_FORMAT_HPP

#include <cstdint>
#include <cstring>

namespace SEM {

// =============================================================================
// SU Trace Header
// =============================================================================

#pragma pack(push, 1)

/**
 * @brief SU trace header (240 bytes, matching SEGY/SU specification)
 *
 * Field byte positions follow the SEG Y standard (Barry et al., 1975)
 * with CWP local extensions at bytes 181-240.
 */
struct SUTraceHeader {
    int32_t  tracl;        ///< bytes 1-4:   trace sequence number within line
    int32_t  tracr;        ///< bytes 5-8:   trace sequence number within file
    int32_t  fldr;         ///< bytes 9-12:  original field record number
    int32_t  tracf;        ///< bytes 13-16: trace number within field record
    int32_t  ep;           ///< bytes 17-20: energy source point number
    int32_t  cdp;          ///< bytes 21-24: ensemble number
    int32_t  cdpt;         ///< bytes 25-28: trace number within ensemble
    int16_t  trid;         ///< bytes 29-30: trace identification code
    int16_t  nvs;          ///< bytes 31-32: number of vertically summed traces
    int16_t  nhs;          ///< bytes 33-34: number of horizontally summed traces
    int16_t  duse;         ///< bytes 35-36: data use (1=production, 2=test)
    int32_t  offset;       ///< bytes 37-40: source-receiver distance
    int32_t  gelev;        ///< bytes 41-44: receiver group elevation
    int32_t  selev;        ///< bytes 45-48: surface elevation at source
    int32_t  sdepth;       ///< bytes 49-52: source depth below surface
    int32_t  gdel;         ///< bytes 53-56: datum elevation at receiver group
    int32_t  sdel;         ///< bytes 57-60: datum elevation at source
    int32_t  swdep;        ///< bytes 61-64: water depth at source
    int32_t  gwdep;        ///< bytes 65-68: water depth at receiver group
    int16_t  scalel;       ///< bytes 69-70: scalar for elevation fields
    int16_t  scalco;       ///< bytes 71-72: scalar for coordinate fields
    int32_t  sx;           ///< bytes 73-76: source coordinate X
    int32_t  sy;           ///< bytes 77-80: source coordinate Y
    int32_t  gx;           ///< bytes 81-84: receiver coordinate X
    int32_t  gy;           ///< bytes 85-88: receiver coordinate Y
    int16_t  counit;       ///< bytes 89-90: coordinate units (1=m, 2=arcsec)
    int16_t  wevel;        ///< bytes 91-92: weathering velocity
    int16_t  swevel;       ///< bytes 93-94: subweathering velocity
    int16_t  sut;          ///< bytes 95-96: uphole time at source (ms)
    int16_t  gut;          ///< bytes 97-98: uphole time at receiver (ms)
    int16_t  sstat;        ///< bytes 99-100: source static correction (ms)
    int16_t  gstat;        ///< bytes 101-102: group static correction (ms)
    int16_t  tstat;        ///< bytes 103-104: total static applied (ms)
    int16_t  laga;         ///< bytes 105-106: lag time A (ms)
    int16_t  lagb;         ///< bytes 107-108: lag time B (ms)
    int16_t  delrt;        ///< bytes 109-110: delay recording time (ms)
    int16_t  muts;         ///< bytes 111-112: mute time start (ms)
    int16_t  mute;         ///< bytes 113-114: mute time end (ms)
    uint16_t ns;           ///< bytes 115-116: number of samples in trace
    uint16_t dt;           ///< bytes 117-118: sample interval (microseconds)
    int16_t  gain;         ///< bytes 119-120: gain type of field instruments
    int16_t  igc;          ///< bytes 121-122: instrument gain constant
    int16_t  igi;          ///< bytes 123-124: instrument initial gain
    int16_t  corr;         ///< bytes 125-126: correlated (1=no, 2=yes)
    int16_t  sfs;          ///< bytes 127-128: sweep frequency at start
    int16_t  sfe;          ///< bytes 129-130: sweep frequency at end
    int16_t  slen;         ///< bytes 131-132: sweep length (ms)
    int16_t  styp;         ///< bytes 133-134: sweep type code
    int16_t  stas;         ///< bytes 135-136: sweep trace taper at start (ms)
    int16_t  stae;         ///< bytes 137-138: sweep trace taper at end (ms)
    int16_t  tatyp;        ///< bytes 139-140: taper type
    int16_t  afilf;        ///< bytes 141-142: alias filter frequency
    int16_t  afils;        ///< bytes 143-144: alias filter slope
    int16_t  nofilf;       ///< bytes 145-146: notch filter frequency
    int16_t  nofils;       ///< bytes 147-148: notch filter slope
    int16_t  lcf;          ///< bytes 149-150: low cut frequency
    int16_t  hcf;          ///< bytes 151-152: high cut frequency
    int16_t  lcs;          ///< bytes 153-154: low cut slope
    int16_t  hcs;          ///< bytes 155-156: high cut slope
    int16_t  year;         ///< bytes 157-158: year data recorded
    int16_t  day;          ///< bytes 159-160: day of year
    int16_t  hour;         ///< bytes 161-162: hour of day (24h)
    int16_t  minute;       ///< bytes 163-164: minute of hour
    int16_t  sec;          ///< bytes 165-166: second of minute
    int16_t  timbas;       ///< bytes 167-168: time basis code
    int16_t  trwf;         ///< bytes 169-170: trace weighting factor
    int16_t  grnors;       ///< bytes 171-172: geophone group number of roll switch
    int16_t  grnofr;       ///< bytes 173-174: geophone group number of trace 1
    int16_t  grnlof;       ///< bytes 175-176: geophone group number of last trace
    int16_t  gaps;         ///< bytes 177-178: gap size
    int16_t  otrav;        ///< bytes 179-180: overtravel taper code
    // CWP local assignments (bytes 181-240)
    float    d1;           ///< bytes 181-184: sample spacing for non-seismic data
    float    f1;           ///< bytes 185-188: first sample location
    float    d2;           ///< bytes 189-192: sample spacing between traces
    float    f2;           ///< bytes 193-196: first trace location
    float    ungpow;       ///< bytes 197-200: negative power for compression
    float    unscale;      ///< bytes 201-204: reciprocal of scaling factor
    int32_t  ntr;          ///< bytes 205-208: number of traces
    int16_t  mark;         ///< bytes 209-210: mark selected traces
    int16_t  shortpad;     ///< bytes 211-212: alignment padding
    int16_t  unass[14];    ///< bytes 213-240: unassigned
};

#pragma pack(pop)

static_assert(sizeof(SUTraceHeader) == 240, "SUTraceHeader must be exactly 240 bytes");

}  // namespace SEM

#endif  // SEM_SU_FORMAT_HPP
