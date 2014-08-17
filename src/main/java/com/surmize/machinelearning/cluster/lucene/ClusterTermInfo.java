package com.surmize.machinelearning.cluster.lucene;

import org.apache.mahout.common.RandomUtils;

public class ClusterTermInfo implements Comparable<ClusterTermInfo> {

  private final String term;
  private final int inClusterDF;
  private final int outClusterDF;
  private final double logLikelihoodRatio;

  ClusterTermInfo(String term, int inClusterDF, int outClusterDF, double logLikelihoodRatio) {
    this.term = term;
    this.inClusterDF = inClusterDF;
    this.outClusterDF = outClusterDF;
    this.logLikelihoodRatio = logLikelihoodRatio;
  }

  @Override
  public int hashCode() {
    return term.hashCode() ^ inClusterDF ^ outClusterDF ^ RandomUtils.hashDouble(logLikelihoodRatio);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof ClusterTermInfo)) {
      return false;
    }
    ClusterTermInfo other = (ClusterTermInfo) o;
    return term.equals(other.getTerm())
        && inClusterDF == other.getInClusterDF()
        && outClusterDF == other.getOutClusterDF()
        && logLikelihoodRatio == other.getLogLikelihoodRatio();
  }

  @Override
  public int compareTo(ClusterTermInfo that) {
    int res = Double.compare(that.logLikelihoodRatio, logLikelihoodRatio);
    if (res == 0) {
      res = term.compareTo(that.term);
    }
    return res;
  }

  public int getInClusterDiff() {
    return this.inClusterDF - this.outClusterDF;
  }

  public String getTerm() {
    return term;
  }

  int getInClusterDF() {
    return inClusterDF;
  }

  int getOutClusterDF() {
    return outClusterDF;
  }

  double getLogLikelihoodRatio() {
    return logLikelihoodRatio;
  }
}
