package edu.cpp.mslabisi.model;

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
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

public class RNN {
    private static final Logger LOG = Logger.getLogger(RNN.class.getName());

    private static final int seed = 54321;
    private static final int layer1Size = 156;
    private static final int layer2Size = 156;
    private static final int denseLayerSize = 32;
    private static final double dropout = 0.2;
    private static final int tBPTTLength = 40;
    private static final int nIn = 1;
    private static final int nOut = 1;

    public static MultiLayerNetwork getModel() {
        MultiLayerNetwork model = null;
        try {
            File savedModel = Constants.getModelRsc().getFile();
            if (savedModel.createNewFile()) {
                // configure model
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new RmsProp())
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

                model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(10));
                model.save(Constants.getModelRsc().getFile());
            } else {
                model = MultiLayerNetwork.load(savedModel, true);
            }

        } catch (IOException e) {
            LOG.severe("‼️ Could not create model\n" + e.getMessage());
        }

        return model;
    }
}
