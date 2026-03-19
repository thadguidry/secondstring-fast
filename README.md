Based on prior work from [https://github.com/teamCohen/secondstring](https://github.com/teamCohen/secondstring)

Claude AI assisted with all of the performance enhancements including:

- FastLevenstein — two-row int DP with ThreadLocal row buffers for radius-limited search, same negative-score API as Levenstein, O(n) space, early-exit trimming
- VectorizedJaro - Vectorized Jaro distance using the Java Vector API (JDK 21 incubator). Uses char arrays and boolean masks instead of string operations.
- Benchmark additions — compare original Levenstein, FastLevenstein, and Apache Commons Text Levenshtein in both QuickBenchmark and LargeDatasetBenchmark
  - `./gradlew benchmark`
  - `./gradlew benchmarkCapture`
  - `./gradlew benchmarkDataset -Pdataset=acm_large`
  - `./gradlew benchmarkDatasetCapture -Pdataset=acm_large`
  - `./gradlew benchmarkDataset -Pdataset=acm_large -Pphase=Levenstein`
  - `./gradlew benchmarkDatasetCapture -Pdataset=acm_large -Pphase=Levenstein`
  - `./gradlew quickBenchmark`
  - `./gradlew quickBenchmarkCapture`
  - `./gradlew quickBenchmark -Pphase=Levenstein`
  - `./gradlew quickBenchmarkCapture -Pphase=Levenstein`
- Vendored Apache Commons Text similarity source for benchmark-only comparison lives under `apache-similarity/`
- Tests — a JUnit test class verifying correctness against the original

Benchmark notes:

- Apache Commons Text Levenshtein is benchmarked on lower-cased inputs to align with SecondString Levenstein's case-insensitive behavior.
- The Commons source is vendored only for benchmark and test comparison, not as part of the main library API.
- Saved benchmark captures should go under `out/benchmarks/`, which is already ignored by git.

Large dataset benchmark snapshot (`./gradlew benchmark`):

| Dataset | Original | Fast | Commons | Fast/Orig | Commons/Orig | Commons/Fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `acm_large` | 2476 ms | 331 ms | 388 ms | 7.48x | 6.38x | 1.17x |
| `dblp_large` | 1641 ms | 229 ms | 194 ms | 7.17x | 8.46x | 0.85x |
| `itunes_amazon_tableB_large` | 3437 ms | 438 ms | 512 ms | 7.85x | 6.71x | 1.17x |

Quick benchmark snapshot (`./gradlew quickBenchmark` on `restaurant` subset):

| Original | Fast | Commons | Fast/Orig | Commons/Orig | Commons/Fast |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1040 ms | 159 ms | 193 ms | 6.54x | 5.39x | 1.21x |

How to reproduce:

1. Verify correctness first:

  ```bash
  ./gradlew test
  ```

  This runs the existing `FastLevenstein` tests plus the Commons Text normalization check.

2. Run the quick benchmark:

  ```bash
  ./gradlew quickBenchmark
  ```

  This uses a small `restaurant` subset and is the fastest way to compare original, fast, and Commons Levenshtein. On the current machine it completed in a few seconds.

  To save the output automatically under `out/benchmarks/`, use:

  ```bash
  ./gradlew quickBenchmarkCapture
  ```

  To limit the quick run to one phase, add property `-Pphase=Levenstein` or another phase name such as `JaroWinkler`, `SoftTFIDF`, or `Dictionary`.

3. Run the large dataset benchmark:

  ```bash
  ./gradlew benchmark
  ```

  This runs `acm_large`, `dblp_large`, and `itunes_amazon_tableB_large` through the benchmark driver. On the current machine it completed in about 1 minute 34 seconds.

  To save the output automatically under `out/benchmarks/`, use:

  ```bash
  ./gradlew benchmarkCapture
  ```

   To limit the run to one phase across all large datasets, add `-Pphase=Levenstein` or another phase name such as `JaroWinkler`, `SoftTFIDF`, or `Dictionary`.

4. Re-run a single large dataset if needed:

  ```bash
  ./gradlew benchmarkDataset -Pdataset=acm_large
  ```

  Replace `acm_large` with `dblp_large` or `itunes_amazon_tableB_large` to isolate one dataset.

  To save the output automatically under `out/benchmarks/`, use:

  ```bash
  ./gradlew benchmarkDatasetCapture -Pdataset=acm_large
  ```

  This writes to a dataset-specific file such as `out/benchmarks/benchmark_dataset_acm_large.out`.

5. Re-run a single phase on a single dataset if needed:

  ```bash
  ./gradlew benchmarkDataset -Pdataset=acm_large -Pphase=Levenstein
  ```

  Phase names are matched case-insensitively. Supported values are `JaroWinkler`, `Levenstein`, `SoftTFIDF`, and `Dictionary`.

  To save the output automatically under `out/benchmarks/`, use:

  ```bash
  ./gradlew benchmarkDatasetCapture -Pdataset=acm_large -Pphase=Levenstein
  ```

  This writes to a phase-specific file such as `out/benchmarks/benchmark_dataset_acm_large_Levenstein.out`.
