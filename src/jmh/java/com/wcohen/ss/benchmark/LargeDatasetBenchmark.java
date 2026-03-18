package com.wcohen.ss.benchmark;

import com.wcohen.ss.JaroWinkler;
import com.wcohen.ss.Levenstein;
import com.wcohen.ss.SoftTFIDF;
import com.wcohen.ss.lookup.SoftTFIDFDictionary;
import com.wcohen.ss.fast.FastBagOfTokens;
import com.wcohen.ss.fast.FastJaroWinkler;
import com.wcohen.ss.fast.FastLevenstein;
import com.wcohen.ss.fast.FastSoftTFIDF;
import com.wcohen.ss.fast.FastSoftTFIDFDictionary;
import com.wcohen.ss.api.StringWrapper;
import com.wcohen.ss.BasicStringWrapperIterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class LargeDatasetBenchmark {

    @FunctionalInterface
    private interface PhaseFn {
        void run();
    }

    public static void main(String[] args) throws Exception {
        Map<String, List<String>> ds = QuickBenchmark.loadAllDatasets();

        List<String> targets = new ArrayList<>(Arrays.asList(
                "acm_large",
                "dblp_large",
                "itunes_amazon_tableB_large"
        ));

        if (args.length > 0) {
            targets = Arrays.asList(args);
        }

        for (String name : targets) {
            List<String> strings = ds.get(name);
            if (strings == null || strings.isEmpty()) {
                System.out.println("Skipping missing dataset: " + name);
                continue;
            }

            System.out.println("\n=== Dataset: " + name + " ===");
            printStats(strings);

            runPhase("JaroWinkler", () -> benchJaroWinkler(strings, Math.min(140, strings.size())));
            runPhase("Levenstein", () -> benchLevenstein(strings, Math.min(140, strings.size())));
            runPhase("SoftTFIDF", () -> benchSoftTFIDF(strings, Math.min(90, strings.size())));
            runPhase("Dictionary", () -> benchDictionary(strings, Math.min(1000, strings.size())));
        }
    }

    private static void runPhase(String name, PhaseFn fn) {
        try {
            fn.run();
        } catch (Throwable t) {
            System.out.println(name + " failed: " + t.getClass().getSimpleName() + " - " + t.getMessage());
        }
    }

    private static void printStats(List<String> strings) {
        long total = 0;
        int max = 0;
        for (String s : strings) {
            int len = s.length();
            total += len;
            if (len > max) max = len;
        }
        double avg = strings.isEmpty() ? 0 : (double) total / strings.size();
        System.out.printf("rows=%d avg_len=%.1f max_len=%d%n", strings.size(), avg, max);
    }

    private static void benchLevenstein(List<String> strings, int n) {
        List<String> subset = strings.subList(0, n);

        Levenstein orig = new Levenstein();
        FastLevenstein fast = new FastLevenstein();

        long t1 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += fast.scoreStrings(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        System.out.printf("Levenstein n=%d orig=%dms fast=%dms speedup=%.2fx%n", n, t1, t2,
                (double) t1 / Math.max(1, t2));
    }

    private static void benchJaroWinkler(List<String> strings, int n) {
        List<String> subset = strings.subList(0, n);

        JaroWinkler orig = new JaroWinkler();
        FastJaroWinkler fast = new FastJaroWinkler();

        long t1 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += fast.scoreStrings(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        System.out.printf("JaroWinkler n=%d orig=%dms fast=%dms speedup=%.2fx%n", n, t1, t2,
                (double) t1 / Math.max(1, t2));
    }

    private static void benchSoftTFIDF(List<String> strings, int n) {
        List<String> subset = strings.subList(0, n);

        SoftTFIDF orig = new SoftTFIDF();
        FastSoftTFIDF fast = new FastSoftTFIDF();

        List<StringWrapper> wrapped = new ArrayList<>();
        for (String s : subset) wrapped.add(orig.prepare(s));
        orig.train(new BasicStringWrapperIterator(wrapped.iterator()));
        fast.train(subset.toArray(new String[0]));

        long t1 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = QuickBenchmark.measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                FastBagOfTokens bi = fast.prepareVector(subset.get(i));
                for (int j = i + 1; j < n; j++) {
                    FastBagOfTokens bj = fast.prepareVector(subset.get(j));
                    sum += fast.scoreBags(bi, bj);
                }
            }
            return sum;
        });

        System.out.printf("SoftTFIDF n=%d orig=%dms fast=%dms speedup=%.2fx%n", n, t1, t2,
                (double) t1 / Math.max(1, t2));
    }

    private static void benchDictionary(List<String> strings, int n) {
        List<String> subset = strings.subList(0, n);
        double minScore = 0.5;

        SoftTFIDFDictionary orig = new SoftTFIDFDictionary();
        FastSoftTFIDFDictionary fast = new FastSoftTFIDFDictionary();
        fast.setMaxInvertedIndexSize(500);

        for (int i = 0; i < n; i++) {
            orig.put(subset.get(i), "v" + i);
            fast.put(subset.get(i), "v" + i);
        }
        orig.freeze();
        fast.freeze();

        List<String> queries = new ArrayList<>();
        int step = Math.max(1, n / 100);
        for (int i = 0; i < n && queries.size() < 30; i += step) {
            queries.add(subset.get(i));
        }

        long t1 = QuickBenchmark.measure(() -> {
            int c = 0;
            for (String q : queries) c += orig.lookup(minScore, q);
            return (double) c;
        });

        long t2 = QuickBenchmark.measure(() -> {
            int c = 0;
            for (String q : queries) c += fast.lookup(minScore, q);
            return (double) c;
        });

        System.out.printf("Dictionary n=%d q=%d orig=%dms fast=%dms speedup=%.2fx%n", n, queries.size(), t1, t2,
                (double) t1 / Math.max(1, t2));
    }
}
