package com.surmize.machinelearning.cluster.lucene;

import java.io.File;
import java.util.Collections;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.math.Vector;

public final class ClusterTopTermsMapper {

    private static final Logger log = LoggerFactory.getLogger(ClusterTopTermsMapper.class);
    private final Path seqFileDir;
    private long maxDocs = Long.MAX_VALUE;
    private String termDictionary;
    private int numTopFeatures = 10;

    public ClusterTopTermsMapper(Path seqFileDir, String termDictionary) {
        this.seqFileDir = seqFileDir;
        this.termDictionary = termDictionary;
    }

    public List<ClusterTerms> getClusterTerms() throws Exception {
        Configuration conf = new Configuration();
        List<ClusterTerms> clusterTermsList = new ArrayList<>();
        String[] dictionary = VectorHelper.loadTermDictionary(new File(this.termDictionary));

        Iterable<ClusterWritable> iterable = new SequenceFileDirValueIterable<>(new Path(seqFileDir, "part-*"), PathType.GLOB, conf);
        Iterator<ClusterWritable> iterator = iterable.iterator();
        long result = 0;
        while (result < maxDocs && iterator.hasNext()) {
            ClusterWritable cw = iterator.next();

            ClusterTerms clusterTerms = new ClusterTerms();
            Cluster cluster = cw.getValue();
            clusterTerms.clusterId = cluster.getId();
            clusterTerms.terms = getTopTerms(cw.getValue().getCenter(), dictionary, numTopFeatures);
            clusterTermsList.add(clusterTerms);
            result++;
        }
        return clusterTermsList;
    }

    private List<String> getTopTerms(Vector vector, String[] dictionary, int numTerms) {
        List<String> terms = new ArrayList<>();
        for (Pair<String, Double> item : getTopPairs(vector, dictionary, numTerms)) {
            terms.add(item.getFirst());
            String term = item.getFirst();
        }
        return terms;
    }

    private static Collection<Pair<String, Double>> getTopPairs(Vector vector, String[] dictionary, int numTerms) {
        List<TermIndexWeight> vectorTerms = Lists.newArrayList();

        for (Vector.Element elt : vector.nonZeroes()) {
            vectorTerms.add(new TermIndexWeight(elt.index(), elt.get()));
        }

        // Sort results in reverse order (ie weight in descending order)
        Collections.sort(vectorTerms, new Comparator<TermIndexWeight>() {
            @Override
            public int compare(TermIndexWeight one, TermIndexWeight two) {
                return Double.compare(two.weight, one.weight);
            }
        });

        Collection<Pair<String, Double>> topTerms = Lists.newLinkedList();

        for (int i = 0; i < vectorTerms.size() && i < numTerms; i++) {
            int index = vectorTerms.get(i).index;
            String dictTerm = dictionary[index];
            if (dictTerm == null) {
                log.error("Dictionary entry missing for {}", index);
                continue;
            }
            topTerms.add(new Pair<String, Double>(dictTerm, vectorTerms.get(i).weight));
        }

        return topTerms;
    }


    public String getTermDictionary() {
        return termDictionary;
    }

    public void setTermDictionary(String termDictionary) {
        this.termDictionary = termDictionary;
    }

    public void setNumTopFeatures(int num) {
        this.numTopFeatures = num;
    }

    public int getNumTopFeatures() {
        return this.numTopFeatures;
    }

    private static class TermIndexWeight {

        private final int index;
        private final double weight;

        TermIndexWeight(int index, double weight) {
            this.index = index;
            this.weight = weight;
        }
    }
}
