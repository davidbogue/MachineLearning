package com.surmize.machinelearning.examples;

import com.surmize.machinelearning.cluster.lucene.ClusterTerms;
import com.surmize.machinelearning.cluster.lucene.ClusterTopTermsMapper;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.lucene.Driver;

public class LuceneIndexToKMeansExample {

    public static void main(String args[]) throws Exception {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        String solrIndexDir = "/Users/davidbogue/solr-4.6.1/example/solr/collection1/data/index";
        String solrVectorsFile = "clustering/testdata/solrvectors/out.vec";
        String solrDictFile = "clustering/testdata/solrvectors/dict.txt";
        String inputClustersDir = "clustering/testdata/input-clusters";
        String outputClusterDir = "clustering/output";

        Class measureClazz = SquaredEuclideanDistanceMeasure.class;
        /* EQUIVELENT TO CLI: 
         ./mahout lucene.vector \
         --dir ~/solr-4.6.1/example/solr/collection1/data/index \
         --field text \
         --dictOut ~/MachineLearning/vectorFiles/dict.txt \
         --output ~/MachineLearning/vectorFiles/out.vec \
         */
        Driver luceneDriver = new Driver();
        luceneDriver.setLuceneDir(solrIndexDir);
        luceneDriver.setField("text");
        luceneDriver.setIdField("id");
        luceneDriver.setOutFile(solrVectorsFile);
        luceneDriver.setDictOut(solrDictFile);

        luceneDriver.dumpVectors();

        /* EQUIVELENT TO CLI: 
         ./mahout kmeans \
         --input ~/MachineLearning/vectorFiles/out.vec \
         --clusters ~/MachineLearning/solr-clusters/out/clusters -k 20 \
         --output ~/MachineLearning/solr-clusters/out/ --distanceMeasure \
         org.apache.mahout.common.distance.CosineDistanceMeasure \
         --convergenceDelta 0.001 --overwrite --maxIter 50 --clustering
         */
        Path inputClustersPath = new Path(inputClustersDir);
        Path solrVectorPath = new Path(solrVectorsFile);
        Path clusterOutputPath = new Path(outputClusterDir);
        Path pointsPath = new Path(outputClusterDir + "/" + Cluster.CLUSTERED_POINTS_DIR);

        String measureClass = measureClazz.getName();
        DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
        inputClustersPath = RandomSeedGenerator.buildRandom(new Configuration(), solrVectorPath, inputClustersPath, 15, measure);

        KMeansDriver.run(solrVectorPath, //the directory pathname for input points
                inputClustersPath, // the directory pathname for initial & computed clusters
                clusterOutputPath, //the directory pathname for output points
                0.001, //the convergence delta value
                50, //the maximum number of iterations
                true, //true if points are to be clustered after iterations are completed
                0, //Is a clustering strictness / outlier removal parameter. Its value should be between 0 and 1. Vectors having pdf below this value will not be clustered.
                true);                   //if true execute sequential algorithm
        /* 
         The cluster labeler code did not perform well.  It needs some enhancements.  For now use TopTermMapper which had much better results
         */
        //        ClusterLabeler labeler = new ClusterLabeler(
        //                output,
        //                new Path("clustering/output/" + Cluster.CLUSTERED_POINTS_DIR),
        //                indexFilesPath,
        //                "text",
        //                5,
        //                10);
        //        labeler.setIdField("id");
        //        List<ClusterLabel> labels = labeler.getLabels();
        //        for (ClusterLabel clusterLabel : labels) {
        //            System.out.println("For Cluster: "+clusterLabel.clusterId);
        //            System.out.println("\t Terms");
        //            for (ClusterTermInfo cti : clusterLabel.termInfoList) {
        //                System.out.println("\t "+cti.getTerm());
        //            }
        //        }

        ClusterTopTermsMapper clusterTopTerms = new ClusterTopTermsMapper(inputClustersPath, solrDictFile);
        List<ClusterTerms> clusterTermsList = clusterTopTerms.getClusterTerms();
        for (ClusterTerms clusterTerms : clusterTermsList) {
            System.out.println("Cluster: " + clusterTerms.clusterId);
            for (String term : clusterTerms.terms) {
                System.out.println("\t" + term);
            }
        }
        //Read and print out the cluster points to the console
        try (SequenceFile.Reader reader = new SequenceFile.Reader(fs,
                new Path("clustering/output/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-0"), conf)) {
            IntWritable key = new IntWritable();
            WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
            while (reader.next(key, value)) {
                Vector v = value.getVector();
                String vectorName = "";
                if (v instanceof NamedVector) {
                    vectorName = ((NamedVector) v).getName();
                }
//                System.out.println(vectorName + " belongs to cluster " + key.toString());
            }
        }
    }

}
