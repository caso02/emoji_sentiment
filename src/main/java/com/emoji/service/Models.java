package com.emoji.service;

import ai.djl.Model;

/**
 * Centralized model configuration for emoji sentiment analysis.
 */
public class Models {
    // Constants for model configuration
    public static final String MODEL_NAME = "emoji-text-model"; // without extension
    public static final int INPUT_LENGTH = 100;
    
    /**
     * Creates a new model instance with the standard configuration.
     * @return A new model instance
     */
    public static Model getModel() {
        return Model.newInstance(MODEL_NAME);
    }
}
