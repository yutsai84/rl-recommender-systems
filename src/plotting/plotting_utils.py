from typing import Dict, List
import matplotlib.pyplot as plt

def plot(metric_results: Dict[str, List[float]] , metric_name: str) -> None:
  plt.plot(metric_results[metric_name])
  plt.ylabel(metric_name)
  plt.xlabel("Step")
  plt.title("{} versus Step".format(metric_name))