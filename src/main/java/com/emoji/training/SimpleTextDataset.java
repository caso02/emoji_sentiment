package com.emoji.training;

import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Progress;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class SimpleTextDataset extends RandomAccessDataset {

    private final List<String> texts = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();
    private final List<String> classNames = List.of("POSITIVE", "NEUTRAL", "NEGATIVE");

    protected SimpleTextDataset(Builder builder) throws IOException {
        super(builder);
        loadCsv(builder.dataFilePath);
    }

    private void loadCsv(String csvPath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            boolean header = true;
            while ((line = br.readLine()) != null) {
                if (header) {
                    header = false;
                    continue;
                }

                // Finde das letzte Komma
                int lastComma = line.lastIndexOf(',');
                if (lastComma == -1) continue;

                // Extrahiere Text und Label
                String text = line.substring(0, lastComma).replace("\"", "").trim();
                String label = line.substring(lastComma + 1).replace("\"", "").trim();

                int labelIndex = classNames.indexOf(label);
                if (labelIndex == -1) {
                    System.err.println("Warnung: Unbekanntes Label gefunden: '" + label + "' in Zeile: " + line);
                    continue;
                }

                texts.add(text);
                labels.add(labelIndex);
            }
        }
    }

    @Override
    public Record get(NDManager manager, long index) {
        String text = texts.get((int) index);
        int label = labels.get((int) index);

        float[] encoded = encodeText(text);
        NDArray data = manager.create(encoded, new Shape(encoded.length));
        NDArray labelArr = manager.create(new int[]{label});
        
        NDList dataList = new NDList(data);
        NDList labelList = new NDList(labelArr);
        return new Record(dataList, labelList);
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

    @Override
    public long size() {
        return texts.size();
    }

    @Override
    public void prepare(Progress progress) throws IOException {
        // Bereits in loadCsv implementiert
    }

    @Override
    public long availableSize() {
        return texts.size();
    }

    public List<String> getClassNames() {
        return classNames;
    }

    public static final class Builder extends BaseBuilder<Builder> {
        String dataFilePath;

        public Builder setDataFilePath(String dataFilePath) {
            this.dataFilePath = dataFilePath;
            return self();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public SimpleTextDataset build() throws IOException {
            return new SimpleTextDataset(this);
        }
    }
}
