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
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;

import java.io.IOException;
import java.nio.file.Files;
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
        int batchSize = 32;
        int epoch = 5;

        SimpleTextDataset trainingSet = new SimpleTextDataset.Builder()
                .setDataFilePath(trainCsv)
                .setSampling(batchSize, true)
                .build();

        SimpleTextDataset testSet = new SimpleTextDataset.Builder()
                .setDataFilePath(testCsv)
                .setSampling(batchSize, true)
                .build();

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

        try (Trainer trainer = model.newTrainer(config)) {
            trainer.initialize(new ai.djl.ndarray.types.Shape(100));  // shape muss zum encodeText passen
            EasyTrain.fit(trainer, epoch, trainingSet, testSet);

            model.save(Paths.get(outputPath), "emoji-text-model");

            // Klassenliste speichern
            Files.write(Paths.get(outputPath, "synset.txt"), trainingSet.getClassNames());

            System.out.println("✅ Modell erfolgreich gespeichert unter: " + outputPath);
        }
    }
}
