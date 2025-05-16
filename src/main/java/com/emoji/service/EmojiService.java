package com.emoji.service;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;
import ai.djl.MalformedModelException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Service
public class EmojiService {

    private Predictor<float[], Classifications> predictor;
    private final Map<String, String> emojiMap = Map.of(
            "POSITIVE", "😊",
            "NEGATIVE", "😢",
            "NEUTRAL", "😐"
    );

    
    @PostConstruct
    public void loadModel() {
    try {
        Path modelDir = Paths.get("models/emoji-text");
        Path modelPath = modelDir.resolve("emoji-text-model.pt");
        List<String> labels = Files.readAllLines(modelDir.resolve("synset.txt"));

        Translator<float[], Classifications> translator = new Translator<>() {
            @Override
            public NDList processInput(TranslatorContext ctx, float[] input) {
                NDManager manager = ctx.getNDManager();
                return new NDList(manager.create(input, new Shape(100)));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                return new Classifications(labels, list.singletonOrThrow());
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // Keine Batches
            }
        };

        Criteria<float[], Classifications> criteria = Criteria.builder()
            .setTypes(float[].class, Classifications.class)
            .optModelPath(modelPath)  // oder: .optModelPath(modelDir), wenn model.pt heißen würde
            .optEngine("PyTorch")     // erzwingt DJL-Backend
            .optTranslator(translator)
            .build();

        ZooModel<float[], Classifications> model = ModelZoo.loadModel(criteria);
        predictor = model.newPredictor();

        System.out.println("✅ PyTorch-Modell erfolgreich geladen.");

    } catch (IOException | MalformedModelException | ModelNotFoundException e) {
        e.printStackTrace();
    }

}

    public String analyzeText(String text) {
        if (predictor == null) return "❌";

        try {
            float[] encoded = encodeText(text);
            Classifications result = predictor.predict(encoded);
            String label = result.best().getClassName();
            return emojiMap.getOrDefault(label, "🤔");
        } catch (Exception e) {
            e.printStackTrace();
            return "❌";
        }
    }

    private float[] encodeText(String text) {
        int maxLength = 100;
        float[] result = new float[maxLength];
        int i = 0;
        for (char c : text.toCharArray()) {
            if (i >= maxLength) break;
            result[i++] = Math.min(1f, c / 255f);
        }
        return result;
    }
}
