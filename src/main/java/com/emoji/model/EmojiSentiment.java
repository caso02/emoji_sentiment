package com.emoji.model;

public class EmojiSentiment {
    private String emoji;
    private double positiveProbability;
    private double neutralProbability;
    private double negativeProbability;
    private double occurrenceCount;

    public EmojiSentiment(String emoji, double positive, double neutral, double negative, double count) {
        this.emoji = emoji;
        this.positiveProbability = positive;
        this.neutralProbability = neutral;
        this.negativeProbability = negative;
        this.occurrenceCount = count;
    }

    public String getEmoji() {
        return emoji;
    }

    public double getPositiveProbability() {
        return positiveProbability;
    }

    public double getNeutralProbability() {
        return neutralProbability;
    }

    public double getNegativeProbability() {
        return negativeProbability;
    }

    public double getOccurrenceCount() {
        return occurrenceCount;
    }

    public String getDominantSentiment() {
        if (positiveProbability > neutralProbability && positiveProbability > negativeProbability) {
            return "POSITIVE";
        } else if (negativeProbability > neutralProbability && negativeProbability > positiveProbability) {
            return "NEGATIVE";
        } else {
            return "NEUTRAL";
        }
    }
} 