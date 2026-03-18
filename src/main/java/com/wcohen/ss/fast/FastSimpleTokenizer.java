package com.wcohen.ss.fast;

import com.wcohen.ss.api.Token;
import com.wcohen.ss.api.Tokenizer;
import java.util.*;

public class FastSimpleTokenizer implements Tokenizer {

    public static final FastSimpleTokenizer DEFAULT_TOKENIZER = new FastSimpleTokenizer(true, true);

    private final boolean ignorePunctuation;
    private final boolean ignoreCase;
    private int nextId = 0;
    private final HashMap<String, Token> tokMap = new HashMap<>(4096);

    public FastSimpleTokenizer(boolean ignorePunctuation, boolean ignoreCase) {
        this.ignorePunctuation = ignorePunctuation;
        this.ignoreCase = ignoreCase;
    }

    @Override
    public Token[] tokenize(String input) {
        int estTokens = 1;
        boolean inToken = false;
        for (int i = 0; i < input.length(); i++) {
            if (Character.isWhitespace(input.charAt(i))) {
                if (inToken) { estTokens++; inToken = false; }
            } else {
                inToken = true;
            }
        }

        Token[] tokens = new Token[estTokens];
        int tokenCount = 0;
        StringBuilder buf = new StringBuilder(32);
        int cursor = 0;

        while (cursor < input.length()) {
            char ch = input.charAt(cursor);
            if (Character.isWhitespace(ch)) {
                cursor++;
            } else if (Character.isLetter(ch)) {
                buf.setLength(0);
                while (cursor < input.length() && Character.isLetter(input.charAt(cursor))) {
                    buf.append(input.charAt(cursor));
                    cursor++;
                }
                if (tokenCount >= tokens.length) tokens = Arrays.copyOf(tokens, tokens.length * 2);
                tokens[tokenCount++] = internSomething(buf.toString());
            } else if (Character.isDigit(ch)) {
                buf.setLength(0);
                while (cursor < input.length() && Character.isDigit(input.charAt(cursor))) {
                    buf.append(input.charAt(cursor));
                    cursor++;
                }
                if (tokenCount >= tokens.length) tokens = Arrays.copyOf(tokens, tokens.length * 2);
                tokens[tokenCount++] = internSomething(buf.toString());
            } else {
                if (!ignorePunctuation) {
                    if (tokenCount >= tokens.length) tokens = Arrays.copyOf(tokens, tokens.length * 2);
                    tokens[tokenCount++] = internSomething(String.valueOf(ch));
                }
                cursor++;
            }
        }

        if (tokenCount == tokens.length) return tokens;
        return Arrays.copyOf(tokens, tokenCount);
    }

    private Token internSomething(String s) {
        return intern(ignoreCase ? s.toLowerCase() : s);
    }

    @Override
    public Token intern(String s) {
        Token tok = tokMap.get(s);
        if (tok == null) {
            tok = new FastToken(++nextId, s);
            tokMap.put(s, tok);
        }
        return tok;
    }

    @Override
    public Iterator<Token> tokenIterator() { return tokMap.values().iterator(); }

    @Override
    public int maxTokenIndex() { return nextId; }

    @Override
    public String toString() {
        return "[FastSimpleTokenizer " + ignorePunctuation + ";" + ignoreCase + "]";
    }
}
