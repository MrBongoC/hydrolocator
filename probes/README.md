# Probes

Standalone scripts that prove each external call the pipeline depends on. Stdlib only, no
application imports. Re-run one before changing the integration that copies it.

| Probe | Proves | Auth |
| --- | --- | --- |
| `gcs_sanctsound.py` | NOAA `noaa-passive-bioacoustic` bucket: listing with `matchGlob`, ship-detection CSV, deployment metadata JSON | none |
| `marinecadastre_ais.py` | Daily AIS zip URL pattern, size, CSV header and first rows (range request, no full download) | none |
| `mbari_pacific_sound.py` | `pacific-sound-2khz` S3 listing and WAV header | none |
| `onc_oceans3.py` | Oceans 3.0 API needs `ONC_TOKEN`; prints locations for the Strait of Georgia when set | token |

Recorded patterns (2026-11-25):

- GCS listing: `GET https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o`
  with `prefix`, `delimiter`, `matchGlob`, `fields`, `maxResults`; paginate on `nextPageToken`.
- GCS object: `GET https://storage.googleapis.com/noaa-passive-bioacoustic/<name>`.
- SanctSound ships CSV: header `ISOStartTime,ISOEndTime,Labels`, UTC ISO with `.000Z`.
- SanctSound metadata JSON: `SHAPE` is WKT `POINT (lon lat)`; `START_TIME`/`END_TIME` are
  `02-Aug-2019 19:00:00` (UTC).
- SanctSound audio: 6 h FLAC files, about 600 MB each, UTC start time in the filename.
- SanctSound `hmd`: daily NetCDF, 1-minute hybrid-millidecade spectra, about 4 MB per day.
- MarineCadastre: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<yyyy>/AIS_<yyyy>_<mm>_<dd>.zip`,
  about 360 MB per day, supports range requests.
- MBARI: `https://pacific-sound-2khz.s3.amazonaws.com/<yyyy>/<mm>/MARS-<yyyymmdd>T000000Z-2kHz.wav`,
  518 MB per day, 24-bit.
- ONC: every endpoint returns error 128 without `token`.
