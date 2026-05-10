\# Sem2Edge



Sem2Edge is a semantic-aware temporal framework for Dynamic Link Prediction (DLP) that jointly models semantic evolution and structural dynamics in evolving graphs.



Unlike traditional temporal graph learning methods that primarily focus on structural evolution or treat semantic embeddings as static node features, Sem2Edge models semantic representations as evolving temporal signals that continuously interact with graph structure during prediction.



\## Features



\- Dynamic Link Prediction (DLP)

\- Temporal graph learning

\- Semantic evolution modeling

\- Semantic–structural co-evolution

\- Multi-negative temporal evaluation

\- Transformer-based temporal reasoning

\- Support for BERT and LLaMA embeddings



\## Datasets



The framework is evaluated on:



\- Reddit

\- Enron



\## Repository Structure



```text

Sem2Edge/

├── data/

├── models/

├── train/

├── evaluation/

├── results/

├── figures/



\##Installation



pip install -r requirements.txt



\##Training



python train\_sem2edge.py



\##Evaluation Metrics



AUC

AP

F1-score

Hits@1

Hits@3

Hits@10

MRR



\##Paper



The paper introduces Sem2Edge as a framework for modeling semantic evolution in dynamic graph learning and dynamic link prediction.



\##Citation



Citation information will be added after publication.





