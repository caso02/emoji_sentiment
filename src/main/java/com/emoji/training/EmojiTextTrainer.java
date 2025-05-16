package com.emoji.training;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class EmojiTextTrainer {

    public static void main(String[] args) throws IOException, TranslateException {
        String trainFile = "src/main/resources/data/train.csv";
        String testFile = "src/main/resources/data/test.csv";
        String modelOutput = "models/emoji-text";
        
        trainModel(trainFile, testFile, modelOutput);
    }

    public static void trainModel(String trainCsv, String testCsv, String outputPath) throws IOException, TranslateException {
        System.out.println("Starting model training...");
        System.out.println("Training file: " + trainCsv);
        System.out.println("Testing file: " + testCsv);
        System.out.println("Output path: " + outputPath);
        
        // Create output directory if it doesn't exist
        Path outputDir = Paths.get(outputPath);
        if (!Files.exists(outputDir)) {
            System.out.println("Creating output directory: " + outputDir);
            Files.createDirectories(outputDir);
        }
        
        int batchSize = 32;
        int epoch = 5;

        System.out.println("Loading training dataset...");
        SimpleTextDataset trainingSet = new SimpleTextDataset.Builder()
                .setDataFilePath(trainCsv)
                .setSampling(batchSize, true)
                .build();

        System.out.println("Loading testing dataset...");
        SimpleTextDataset testSet = new SimpleTextDataset.Builder()
                .setDataFilePath(testCsv)
                .setSampling(batchSize, true)
                .build();

        System.out.println("Creating model...");
        Model model = Model.newInstance("emoji-text-model");

        Block block = new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(ai.djl.nn.Activation::relu)
                .add(Linear.builder().setUnits(64).build())
                .add(ai.djl.nn.Activation::relu)
                .add(Linear.builder().setUnits(3).build()); // POSITIVE, NEUTRAL, NEGATIVE

        model.setBlock(block);

        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.01f)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        System.out.println("Starting training...");
        try (Trainer trainer = model.newTrainer(config)) {
            trainer.initialize(new ai.djl.ndarray.types.Shape(100));  // shape muss zum encodeText passen
            EasyTrain.fit(trainer, epoch, trainingSet, testSet);

            // Save model with different formats to ensure compatibility
            System.out.println("Saving model to: " + outputPath);
            model.save(Paths.get(outputPath), "emoji-text-model");
            System.out.println("Model saved successfully as 'emoji-text-model'");
            
            // Also save with .pt extension for PyTorch compatibility
            Path ptModelFile = Paths.get(outputPath, "emoji-text-model.pt");
            if (Files.exists(Paths.get(outputPath, "emoji-text-model-0001.params"))) {
                Files.copy(Paths.get(outputPath, "emoji-text-model-0001.params"), ptModelFile);
                System.out.println("Also saved as PyTorch compatible file: emoji-text-model.pt");
            }

            // Save labels file
            System.out.println("Saving label file to: " + Paths.get(outputPath, "synset.txt"));
            Files.write(Paths.get(outputPath, "synset.txt"), trainingSet.getClassNames());

            // Print success message with clear instructions
            System.out.println("\n✅ Model training and saving complete!");
            System.out.println("Model files saved to: " + outputPath);
            System.out.println("You can now start your Spring Boot application.");
        }
    }
}