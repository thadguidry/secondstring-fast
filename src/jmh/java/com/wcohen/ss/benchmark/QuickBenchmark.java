package com.wcohen.ss.benchmark;

import com.wcohen.ss.JaroWinkler;
import com.wcohen.ss.Levenstein;
import com.wcohen.ss.SoftTFIDF;
import com.wcohen.ss.lookup.SoftTFIDFDictionary;
import com.wcohen.ss.fast.FastJaroWinkler;
import com.wcohen.ss.fast.FastLevenstein;
import com.wcohen.ss.fast.FastSoftTFIDF;
import com.wcohen.ss.fast.FastSoftTFIDFDictionary;
import com.wcohen.ss.fast.FastBagOfTokens;
import com.wcohen.ss.api.StringWrapper;
import com.wcohen.ss.BasicStringWrapperIterator;
import org.apache.commons.text.similarity.LevenshteinDistance;

import java.io.*;
import java.util.*;

public class QuickBenchmark {

    public static void main(String[] args) throws Exception {
        Map<String, List<String>> ds = loadAllDatasets();
        Set<String> selectedPhases = parseSelectedPhases();
        System.out.println("Datasets loaded: " + ds.keySet());
        runPhase(selectedPhases, "JaroWinkler", () -> benchmarkJaroWinkler(ds));
        runPhase(selectedPhases, "Levenstein", () -> benchmarkLevenstein(ds));
        runPhase(selectedPhases, "SoftTFIDF", () -> benchmarkSoftTFIDF(ds));
        runPhase(selectedPhases, "Dictionary", () -> benchmarkDictionaryLookup(ds));
    }

    static Set<String> parseSelectedPhases() {
        String phaseValue = System.getProperty("benchmark.phase");
        if (phaseValue == null || phaseValue.trim().isEmpty()) {
            return Set.of();
        }

        Set<String> selected = new LinkedHashSet<>();
        String[] parts = phaseValue.split(",");
        for (String part : parts) {
            String normalized = normalizePhaseName(part);
            if (normalized != null) {
                selected.add(normalized);
            }
        }

        if (selected.isEmpty()) {
            throw new IllegalArgumentException(
                "Invalid benchmark.phase='" + phaseValue + "'. Valid values: JaroWinkler, Levenstein, SoftTFIDF, Dictionary"
            );
        }
        return selected;
    }

    static String normalizePhaseName(String rawValue) {
        if (rawValue == null) {
            return null;
        }

        String compact = rawValue.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
        return switch (compact) {
            case "jarowinkler", "jaro" -> "JaroWinkler";
            case "levenstein", "levenshtein", "lev" -> "Levenstein";
            case "softtfidf", "softtf", "tfidf" -> "SoftTFIDF";
            case "dictionary", "dict", "lookup" -> "Dictionary";
            default -> null;
        };
    }

    static void runPhase(Set<String> selectedPhases, String name, Runnable fn) {
        if (!selectedPhases.isEmpty() && !selectedPhases.contains(name)) {
            return;
        }
        fn.run();
    }

    static void benchmarkJaroWinkler(Map<String, List<String>> datasets) {
        List<String> strings = datasets.getOrDefault("restaurant", datasets.values().iterator().next());
        int n = Math.min(200, strings.size());
        List<String> subset = strings.subList(0, n);

        JaroWinkler orig = new JaroWinkler();
        FastJaroWinkler fast = new FastJaroWinkler();

        long t1 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += fast.scoreStrings(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        System.out.println("JaroWinkler original ms: " + t1);
        System.out.println("JaroWinkler fast ms:     " + t2);
        System.out.printf("JaroWinkler speedup: %.2fx%n", (double) t1 / Math.max(1, t2));
    }

    static void benchmarkLevenstein(Map<String, List<String>> datasets) {
        List<String> strings = datasets.getOrDefault("restaurant", datasets.values().iterator().next());
        int n = Math.min(200, strings.size());
        List<String> subset = strings.subList(0, n);
        List<String> normalizedSubset = caseFold(subset);

        Levenstein orig = new Levenstein();
        FastLevenstein fast = new FastLevenstein();
        LevenshteinDistance commons = LevenshteinDistance.getDefaultInstance();

        long t1 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += fast.scoreStrings(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t3 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += commons.apply(normalizedSubset.get(i), normalizedSubset.get(j));
                }
            }
            return sum;
        });

        System.out.println("Levenstein original ms: " + t1);
        System.out.println("Levenstein fast ms:     " + t2);
        System.out.println("Levenstein commons ms:  " + t3);
        System.out.printf("Levenstein fast/orig speedup: %.2fx%n", (double) t1 / Math.max(1, t2));
        System.out.printf("Levenstein commons/orig speedup: %.2fx%n", (double) t1 / Math.max(1, t3));
        System.out.printf("Levenstein commons/fast speedup: %.2fx%n", (double) t3 / Math.max(1, t2));
    }

    static void benchmarkSoftTFIDF(Map<String, List<String>> datasets) {
        List<String> strings = datasets.getOrDefault("restaurant", datasets.values().iterator().next());
        int n = Math.min(100, strings.size());
        List<String> subset = strings.subList(0, n);

        SoftTFIDF orig = new SoftTFIDF();
        FastSoftTFIDF fast = new FastSoftTFIDF();

        List<StringWrapper> wrapped = new ArrayList<>();
        for (String s : subset) wrapped.add(orig.prepare(s));
        orig.train(new BasicStringWrapperIterator(wrapped.iterator()));

        fast.train(subset.toArray(new String[0]));

        long t1 = measure(() -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    sum += orig.score(subset.get(i), subset.get(j));
                }
            }
            return sum;
        });

        long t2 = measure(() -> {
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

        System.out.println("SoftTFIDF original ms: " + t1);
        System.out.println("SoftTFIDF fast ms:     " + t2);
        System.out.printf("SoftTFIDF speedup: %.2fx%n", (double) t1 / Math.max(1, t2));
    }

    static void benchmarkDictionaryLookup(Map<String, List<String>> datasets) {
        List<String> strings = datasets.getOrDefault("restaurant", datasets.values().iterator().next());
        int n = Math.min(1000, strings.size());
        List<String> subset = strings.subList(0, n);
        double minScore = 0.5;

        SoftTFIDFDictionary orig = new SoftTFIDFDictionary();
        FastSoftTFIDFDictionary fast = new FastSoftTFIDFDictionary();
        fast.setMaxInvertedIndexSize(400);
        for (int i = 0; i < n; i++) {
            orig.put(subset.get(i), "v" + i);
            fast.put(subset.get(i), "v" + i);
        }
        orig.freeze();
        fast.freeze();

        List<String> queries = new ArrayList<>();
        int step = Math.max(1, n / 50);
        for (int i = 0; i < n && queries.size() < 50; i += step) queries.add(subset.get(i));

        long t1 = measure(() -> {
            int c = 0;
            for (String q : queries) c += orig.lookup(minScore, q);
            return (double) c;
        });

        long t2 = measure(() -> {
            int c = 0;
            for (String q : queries) c += fast.lookup(minScore, q);
            return (double) c;
        });

        System.out.println("Dictionary lookup original ms: " + t1);
        System.out.println("Dictionary lookup fast ms:     " + t2);
        System.out.printf("Dictionary lookup speedup: %.2fx%n", (double) t1 / Math.max(1, t2));
    }

    static Map<String, List<String>> loadAllDatasets() throws IOException {
        File dataDir = new File("data");
        if (!dataDir.exists()) dataDir = new File("/root/secondstring-fast/data");
        if (!dataDir.exists()) dataDir = new File("/root/secondstring/data");

        Map<String, List<String>> out = new LinkedHashMap<>();
        File[] files = dataDir.listFiles((d, n) -> n.endsWith(".txt") && !n.equals("README.txt"));
        if (files == null) return out;

        for (File f : files) {
            List<String> items = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split("\\t");
                    if (parts.length >= 3) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = 2; i < parts.length; i++) {
                            if (sb.length() > 0) sb.append(' ');
                            sb.append(parts[i]);
                        }
                        items.add(sb.toString());
                    } else if (parts.length >= 1) {
                        items.add(parts[parts.length - 1]);
                    }
                }
            }
            out.put(f.getName().replace(".txt", ""), items);
        }
        return out;
    }

    static List<String> caseFold(List<String> strings) {
        List<String> normalized = new ArrayList<>(strings.size());
        for (String value : strings) {
            normalized.add(value.toLowerCase(Locale.ROOT));
        }
        return normalized;
    }

    @FunctionalInterface
    interface BenchFn { double run(); }

    static long measure(BenchFn fn) {
        for (int i = 0; i < 3; i++) fn.run();
        long start = System.nanoTime();
        double sink = fn.run();
        long end = System.nanoTime();
        if (sink == 42.424242) System.out.println("sink");
        return (end - start) / 1_000_000;
    }
}
