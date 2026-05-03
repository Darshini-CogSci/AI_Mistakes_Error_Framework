# AI_Mistakes_Error_Framework
An attempt to look closely at errors made by CNN Resnet50 to analyse its severity, depth and source. This work builds upon the works of Geirhos et al. (2018) and Tartaglini et al. (2022) to analyze errors made by CNN in different novel images and how it is influenced by the cues present or absent.

## How to use:
Store the model weights of Resnet50 in your local device.

## Code structure
1. Use the code in stimuli folder to generate novel stimuli
2. Use resnet50_response_trial_data to collect model inference and top broad response label for analysis.
3. Use extract_embeddings to collect the embeddings of each image in a condition from selective layers of the model.
4. Use cos_sim_distinctiveness to verify the distinctiveness of the global shape of a category.
5. Use nna_ambiguity to visualize model confusion in identifying all image sof a category.
6. Use the files in tsne of different layers folder to visualize the clustering of embeddings condition-wise for the model.
For human data and other manipulations done by the authors, please visit the works in the reference section.
## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/Darshini-CogSci/AI_Mistakes.git](https://github.com/Darshini-CogSci/AI_Mistakes.git)
2. Setup the environment
   ```bash
   conda env create -f environment.yml
   conda activate shape-bias
4. Install the dependencies
   ```bash
   pip install -r requirements.txt

## References

- Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2018).** *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* arXiv (Cornell University). [https://doi.org/10.48550/arxiv.1811.12231](https://doi.org/10.48550/arxiv.1811.12231)
- Tartaglini, A. R., Vong, W. K., & Lake, B. M. (2022).** *A Developmentally-Inspired Examination of Shape versus Texture Bias in Machines.* arXiv (Cornell University). [https://doi.org/10.48550/arxiv.2202.08340](https://doi.org/10.48550/arxiv.2202.08340)
