Based on prior work from [https://github.com/teamCohen/secondstring](https://github.com/teamCohen/secondstring)

Claude AI assisted with all of the performance enhancements including:

- FastLevenstein — two-row int DP with ThreadLocal row buffers for radius-limited search, same negative-score API as Levenstein, O(n) space, early-exit trimming
- VectorizedJaro - Vectorized Jaro distance using the Java Vector API (JDK 21 incubator). Uses char arrays and boolean masks instead of string operations.
- Benchmark additions — add Levenstein vs FastLevenstein phase to both QuickBenchmark and LargeDatasetBenchmark
- Tests — a JUnit test class verifying correctness against the original
