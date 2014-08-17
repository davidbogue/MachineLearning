package com.surmize.machinelearning.cluster.lucene;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import java.util.ArrayList;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocsEnum;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.OpenBitSet;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.mahout.utils.vectors.TermEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Get labels for the cluster using Log Likelihood Ratio (LLR).
 * <p/>
 *"The most useful way to think of this (LLR) is as the percentage of in-cluster documents that have the
 * feature (term) versus the percentage out, keeping in mind that both percentages are uncertain since we have
 * only a sample of all possible documents." - Ted Dunning
 * <p/>
 * More about LLR can be found at : http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html
 */
public class ClusterLabeler {

  private static final Logger log = LoggerFactory.getLogger(ClusterLabeler.class);

  public static final int DEFAULT_MIN_IDS = 50;
  public static final int DEFAULT_MAX_LABELS = 25;

  private final String indexDir;
  private final String contentField;
  private String idField;
  private final Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints;
  private final int minNumIds;
  private final int maxLabels;

  public ClusterLabeler(Path seqFileDir,
                       Path pointsDir,
                       String indexDir,
                       String contentField,
                       int minNumIds,
                       int maxLabels) {
    this.indexDir = indexDir;
    this.contentField = contentField;
    this.minNumIds = minNumIds;
    this.maxLabels = maxLabels;
    ClusterDumper clusterDumper = new ClusterDumper(seqFileDir, pointsDir);
    this.clusterIdToPoints = clusterDumper.getClusterIdToPoints();
  }

  public List<ClusterLabel> getLabels() throws IOException {
    List<ClusterLabel> labels = new ArrayList<ClusterLabel>();
    for (Map.Entry<Integer, List<WeightedPropertyVectorWritable>> integerListEntry : clusterIdToPoints.entrySet()) {
      List<WeightedPropertyVectorWritable> wpvws = integerListEntry.getValue();
      List<ClusterTermInfo> termInfos = getClusterLabels(integerListEntry.getKey(), wpvws);
      if (termInfos != null) {
        ClusterLabel label = new ClusterLabel();
        label.clusterId = integerListEntry.getKey();
        label.termInfoList = termInfos;
        labels.add(label);
      }
    }
    return labels;
  }

  /**
   * Get the list of labels, sorted by best score.
   */
  protected List<ClusterTermInfo> getClusterLabels(Integer integer,
                                                        Collection<WeightedPropertyVectorWritable> wpvws) throws IOException {

    if (wpvws.size() < minNumIds) {
      log.info("Skipping small cluster {} with size: {}", integer, wpvws.size());
      return null;
    }

    log.info("Processing Cluster {} with {} documents", integer, wpvws.size());
    Directory dir = FSDirectory.open(new File(this.indexDir));
    IndexReader reader = DirectoryReader.open(dir);
    
    
    log.info("# of documents in the index {}", reader.numDocs());

    Collection<String> idSet = Sets.newHashSet();
    for (WeightedPropertyVectorWritable wpvw : wpvws) {
      Vector vector = wpvw.getVector();
      if (vector instanceof NamedVector) {
        idSet.add(((NamedVector) vector).getName());
      }
    }

    int numDocs = reader.numDocs();

    OpenBitSet clusterDocBitset = getClusterDocBitset(reader, idSet, this.idField);

    log.info("Populating term infos from the index");

    /**
     * This code is as that of CachedTermInfo, with one major change, which is to get the document frequency.
     * 
     * Since we have deleted the documents out of the cluster, the document frequency for a term should only
     * include the in-cluster documents. The document frequency obtained from TermEnum reflects the frequency
     * in the entire index. To get the in-cluster frequency, we need to query the index to get the term
     * frequencies in each document. The number of results of this call will be the in-cluster document
     * frequency.
     */
    Terms t = MultiFields.getTerms(reader, contentField);
    TermsEnum te = t.iterator(null);
    Map<String, TermEntry> termEntryMap = new LinkedHashMap<String, TermEntry>();
    Bits liveDocs = MultiFields.getLiveDocs(reader); //WARNING: returns null if there are no deletions


    int count = 0;
    BytesRef term;
    while ((term = te.next()) != null) {
      OpenBitSet termBitset = new OpenBitSet(reader.maxDoc());
      DocsEnum docsEnum = MultiFields.getTermDocsEnum(reader, null, contentField, term);
      int docID;
      while ((docID = docsEnum.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
        //check to see if we don't have an deletions (null) or if document is live
        if (liveDocs != null && !liveDocs.get(docID)) {
          // document is deleted...
          termBitset.set(docsEnum.docID());
        }
      }
      // AND the term's bitset with cluster doc bitset to get the term's in-cluster frequency.
      // This modifies the termBitset, but that's fine as we are not using it anywhere else.
      termBitset.and(clusterDocBitset);
      int inclusterDF = (int) termBitset.cardinality();

      TermEntry entry = new TermEntry(term.utf8ToString(), count++, inclusterDF);
      termEntryMap.put(entry.getTerm(), entry);

    }

    List<ClusterTermInfo> clusteredTermInfo = Lists.newLinkedList();

    int clusterSize = wpvws.size();

    for (TermEntry termEntry : termEntryMap.values()) {
        
      int corpusDF = reader.docFreq(new Term(this.contentField,termEntry.getTerm()));
      int outDF = corpusDF - termEntry.getDocFreq();
      int inDF = termEntry.getDocFreq();
      double logLikelihoodRatio = scoreDocumentFrequencies(inDF, outDF, clusterSize, numDocs);
      ClusterTermInfo termInfoCluster =
          new ClusterTermInfo(termEntry.getTerm(), inDF, outDF, logLikelihoodRatio);
      clusteredTermInfo.add(termInfoCluster);
    }

    Collections.sort(clusteredTermInfo);
    // Cleanup
    Closeables.close(reader, true);
    termEntryMap.clear();

    return clusteredTermInfo.subList(0, Math.min(clusteredTermInfo.size(), maxLabels));
  }

  private static OpenBitSet getClusterDocBitset(IndexReader reader,
                                                Collection<String> idSet,
                                                String idField) throws IOException {
    int numDocs = reader.numDocs();

    OpenBitSet bitset = new OpenBitSet(numDocs);
    
    Set<String>  idFieldSelector = null;
    if (idField != null) {
      idFieldSelector = new TreeSet<String>();
      idFieldSelector.add(idField);
    }
    
    
    for (int i = 0; i < numDocs; i++) {
      String id;
      // Use Lucene's internal ID if idField is not specified. Else, get it from the document.
      if (idField == null) {
        id = Integer.toString(i);
      } else {
        id = reader.document(i, idFieldSelector).get(idField);
      }
      if (idSet.contains(id)) {
        bitset.set(i);
      }
    }
    log.info("Created bitset for in-cluster documents : {}", bitset.cardinality());
    return bitset;
  }

  private static double scoreDocumentFrequencies(long inDF, long outDF, long clusterSize, long corpusSize) {
    long k12 = clusterSize - inDF;
    long k22 = corpusSize - clusterSize - outDF;

    if (k22<0){ return 0d; }
    return LogLikelihood.logLikelihoodRatio(inDF, k12, outDF, k22);
  }

  public String getIdField() {
    return idField;
  }

  public void setIdField(String idField) {
    this.idField = idField;
  }

}
