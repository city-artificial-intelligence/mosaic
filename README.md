## MOSAIC: Multi-strategy Ontology Alignment and Integration Suite at CSG

A system created as a combination of matching strategies developed by BSc and MSc students taking the module on [Semantic Web Technologies and Knowledge Graphs](https://github.com/turing-knowledge-graphs/teaching/tree/main/city) at City St George's, University of London.


# -- instructions --

## dependencies

### for windows NVIDIA users, download the repository and then create a virtual environment. Then use the terminal/CMD in the root folder to run pip install [paste the list in the requirements.txt file ignoring the commented out lines]
### (or type pip install -r requirements.txt if that works for you)

### for AMD users, use the amd_installation.md


## formatting
### tracks in DH/Bio-ML tested in rdf format - tracks with duplicate ontologies like pactols need to be named in the format pactols1 pactols2 etc.
### reference files can be in .tsv format or .ttl
### rename the source and target rdf files to their exact name: eg source.rdf -> idai.rdf used for the reference.rdf file idai.pactols1.rdf 
### the reference file should be the exact names of the 2 ontologies combined with a "-" in between: eg the format defc-pactols1.rdf or idai-pactols2.rdf
### the rdf format was used because the track has different versions of the same ontology per benchmark

## eval knowledge-graph track
### use kg_eval.py after running amdCode.py to get the correct results saved to kg_eval_report.csv. mosaic_report.csv is the main csv file for the other scores.

