package com.emoji.service;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Service
public class EmojiService {

    private static final Logger logger = LoggerFactory.getLogger(EmojiService.class);
    private Predictor<float[], Classifications> predictor;
    private final Map<String, String> emojiMap = Map.of(
            "POSITIVE", "😊",
            "NEGATIVE", "😢",
            "NEUTRAL", "😐"
    );
    
    @PostConstruct
    public void loadModel() {
        try {
            logger.info("Starting to load emoji sentiment model");
            
            // Create model instance using Models helper class
            logger.debug("Creating model instance via Models.getModel()");
            Model model = Models.getModel();
            logger.debug("Model instance created successfully");
            
            // Path to model directory
            Path modelDir = Paths.get("models/emoji-text");
            logger.info("Model directory: {}", modelDir.toAbsolutePath());
            logger.info("Model name: {}", Models.MODEL_NAME);
            
            // List all files in directory for debugging
            logger.info("Files in model directory:");
            Files.list(modelDir).forEach(path -> {
                logger.info("- {}", path.getFileName());
            });
            
            // Load the labels
            Path synsetPath = modelDir.resolve("synset.txt");
            List<String> labels;
            if (Files.exists(synsetPath)) {
                labels = Files.readAllLines(synsetPath);
                logger.info("Loaded labels: {}", labels);
            } else {
                labels = List.of("POSITIVE", "NEUTRAL", "NEGATIVE");
                logger.warn("Synset file not found. Using default labels: {}", labels);
            }
            
            // Create translator for processing input/output
            Translator<float[], Classifications> translator = new Translator<>() {
                @Override
                public NDList processInput(TranslatorContext ctx, float[] input) {
                    NDManager manager = ctx.getNDManager();
                    return new NDList(manager.create(input, new Shape(Models.INPUT_LENGTH)));
                }

                @Override
                public Classifications processOutput(TranslatorContext ctx, NDList list) {
                    return new Classifications(labels, list.singletonOrThrow());
                }

                @Override
                public Batchifier getBatchifier() {
                    return null; // No batches
                }
            };
            
            // Load the model
            logger.debug("Loading model from directory: {} with name: {}", modelDir, Models.MODEL_NAME);
            model.load(modelDir, Models.MODEL_NAME);
            logger.info("Model successfully loaded");
            
            // Create predictor
            predictor = model.newPredictor(translator);
            logger.info("Predictor successfully initialized");
            
        } catch (Exception e) {
            logger.error("Failed to load model: {}", e.getMessage(), e);
        }
    }

    public String analyzeText(String text) {
        if (predictor == null) {
            logger.error("Predictor is not initialized");
            return "❌";
        }

        try {
            float[] encoded = encodeText(text);
            Classifications result = predictor.predict(encoded);
            String label = result.best().getClassName();
            logger.info("Analyzed text: '{}', Result: {}", text, label);
            return emojiMap.getOrDefault(label, "🤔");
        } catch (Exception e) {
            logger.error("Error analyzing text: {}", e.getMessage(), e);
            return "❌";
        }
    }

    private float[] encodeText(String text) {
        int maxLength = Models.INPUT_LENGTH;
        float[] result = new float[maxLength];
        int i = 0;
        for (char c : text.toCharArray()) {
            if (i >= maxLength) break;
            result[i++] = Math.min(1f, c / 255f);
        }
        return result;
    }
}