package edu.cpp.mslabisi.predict;

import edu.cpp.mslabisi.data.Constants;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.logging.Logger;

public class RNN {
    private static final Logger LOG = Logger.getLogger(RNN.class.getName());

    private static final int seed = 54321;
    private static final int layer1Size = 156;
    private static final int layer2Size = 156;
    private static final int denseLayerSize = 32;
    private static final double dropout = 0.25;
    private static final double learningRate = 0.005;
    private static final int tBPTTLength = 40;
    private static final int nIn = 1;
    private static final int nOut = 1;

    public static MultiLayerNetwork getModel() {
        MultiLayerNetwork model = null;
        if (!Constants.getModelRsc().exists()) {
            // configure model
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(learningRate))
                    .list()
                    .layer(0, new LSTM.Builder()
                            .nIn(nIn)
                            .nOut(layer1Size)
                            .activation(Activation.TANH)
                            .gateActivationFunction(Activation.HARDSIGMOID)
                            .dropOut(dropout)
                            .build())
                    .layer(1, new LSTM.Builder()
                            .nIn(layer1Size)
                            .nOut(layer2Size)
                            .activation(Activation.TANH)
                            .gateActivationFunction(Activation.HARDSIGMOID)
                            .dropOut(dropout)
                            .build())
                    .layer(2, new DenseLayer.Builder()
                            .nIn(layer2Size)
                            .nOut(denseLayerSize)
                            .activation(Activation.RELU)
                            .build())
                    .layer(3, new RnnOutputLayer.Builder()
                            .nIn(denseLayerSize)
                            .nOut(nOut)
                            .activation(Activation.IDENTITY)
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .build())
                    .backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(tBPTTLength)
                    .tBPTTBackwardLength(tBPTTLength)
                    .build();

            // build model
            model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));
            saveModel(model);
        } else {
            model = restoreModel();
        }

//        ComputationGraph model = null;
//        if (!Constants.getModelRsc().exists()) {
//            // configure model
//            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                    .seed(seed)
//                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                    .weightInit(WeightInit.XAVIER)
//                    .updater(new Adam(learningRate))
//                    .list()
//                    .layer(0, new LSTM.Builder()
//                            .nIn(nIn)
//                            .nOut(layer1Size)
//                            .activation(Activation.TANH)
//                            .gateActivationFunction(Activation.HARDSIGMOID)
//                            .dropOut(dropout)
//                            .build())
//                    .layer(1, new LSTM.Builder()
//                            .nIn(layer1Size)
//                            .nOut(layer2Size)
//                            .activation(Activation.TANH)
//                            .gateActivationFunction(Activation.HARDSIGMOID)
//                            .dropOut(dropout)
//                            .build())
//                    .layer(2, new DenseLayer.Builder()
//                            .nIn(layer2Size)
//                            .nOut(denseLayerSize)
//                            .activation(Activation.RELU)
//                            .build())
//                    .layer(3, new RnnOutputLayer.Builder()
//                            .nIn(denseLayerSize)
//                            .nOut(nOut)
//                            .activation(Activation.IDENTITY)
//                            .lossFunction(LossFunctions.LossFunction.MSE)
//                            .build())
//                    .backpropType(BackpropType.TruncatedBPTT)
//                    .tBPTTForwardLength(tBPTTLength)
//                    .tBPTTBackwardLength(tBPTTLength)
//                    .build();
//
//            // build model
//            model = new MultiLayerNetwork(conf);
//            model.init();
//            model.setListeners(new ScoreIterationListener(10));
//            saveModel(model);
//        } else {
//            model = restoreModel();
//        }
//
        return model;
    }

    public static void saveModel(MultiLayerNetwork model) {
        try {
            model.save(Constants.getModelRsc(), true);
        } catch (IOException e) {
            LOG.severe("‼️ Could not save LSTM model\n" + e.getMessage());
        }
    }

    public static MultiLayerNetwork restoreModel() {
        try {
            return MultiLayerNetwork.load(Constants.getModelRsc(), true);
        } catch (IOException e) {
            LOG.severe("‼️ Could not restore LSTM model\n" + e.getMessage());
        }
        return null;
    }
}
