import org.apache.uima.resource.ResourceInitializationException;
import org.canova.nlp.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Created by ako on 1/13/2017.
 */

public class Test2BagOfWords {

    private BagOfWordsVectorizer createBowVectorizer() {
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        ClassPathResource resource = new ClassPathResource("/test2_sentences.txt");
        LabelAwareListSentenceIterator iter = new LabelAwareListSentenceIterator(resource.getInputStream(), ",");

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        /*
         * Build words vectorizer
         */
        BagOfWordsVectorizer bowv = new BagOfWordsVectorizer.Builder()
                .setTokenizerFactory(t)
                .setIterator(iter)
                .setMinWordFrequency(1)
                .setVocab(cache)
                .build();

        bowv.fit();
        Collection<VocabWord> words = cache.vocabWords();
        for (
                VocabWord word : words)

        {
            info(String.format("%d = %s", word.getIndex(), word.getWord()));
        }

    }

    @Test
    public void testTrainBagOfWords() throws ResourceInitializationException, IOException {

        BagOfWordsVectorizer bowv = createBowVectorizer();
        INDArray ar = bowv.transform("What do you see?");
        info(ar.toString());

        /*
         * Build list of training data sets
         */
        ClassPathResource resource = new ClassPathResource("/test2_sentences.txt");
        LabelAwareListSentenceIterator iter = new LabelAwareListSentenceIterator(resource.getInputStream(), ",");

        iter.reset();
        List<DataSet> datasets = new ArrayList<DataSet>();
        int rows = 0;
        while (iter.hasNext()) {
            String sentence = iter.nextSentence();
            String label = iter.currentLabel();
            info(String.format("%s - %s", label, sentence));
            DataSet ds = bowv.vectorize(sentence, label);
            datasets.add(ds);
        }
//        info(words.toString());

        /*
         * Build net model
         */

        int noInputs = datasets.get(0).numInputs();
        int noOutputs = datasets.get(0).numOutcomes();
        int noHidden = (noInputs + noOutputs) / 2;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(1000)
                .learningRate(0.1)
                .useDropConnect(false)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .dist(new UniformDistribution(0, 1))
                .biasInit(0)
                .miniBatch(false)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(noInputs).nOut(noHidden)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
//                        .updater(Updater.SGD)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(noHidden).nOut(noOutputs)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .build())
                .pretrain(true)
                .backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < 10; i++) {
            for (DataSet ds : datasets) {
                net.fit(ds);
            }
        }
        // show result for train data
        for (DataSet ds : datasets) {
            info(ds.getFeatures().toString());
            INDArray output = net.output(ds.getFeatureMatrix());
            info(output.toString());

            Evaluation eval = new Evaluation(2);
            eval.eval(ds.getLabels(), output);
            info(eval.stats());
        }
        /*
         * Let's try
         */
        int[] result = net.predict(bowv.transform("Can you give me some help?"));
        info(Arrays.toString(result));

        /*
         * Save trained network
         */
        File locationToSave = new File("c:/temp/bow-net.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

    }

    @Test
    public void testUseBagOfWords() throws IOException {

        /*
         * Use training set vectorizer
         */
        BagOfWordsVectorizer bowv = createBowVectorizer();

        //Load the model
        File locationToSave = new File("c:/temp/bow-net.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        /*
         * Lets try on training set
         */
        ClassPathResource resource = new ClassPathResource("/test2_sentences.txt");
        LabelAwareListSentenceIterator iter = new LabelAwareListSentenceIterator(resource.getInputStream(), ",");

        iter.reset();
        while (iter.hasNext()) {
            String sentence = iter.nextSentence();
            String label = iter.currentLabel();
            info(String.format("Testing %s = %s", sentence, label));

        }


    }

    private void info(String a) {
        System.out.println(a);
    }
}
