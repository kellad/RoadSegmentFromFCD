#!/usr/bin/env python
# coding: utf-8

# # Node classification with Graph Convolutional Network (GCN)
# 
# > This demo explains how to do node classification using the StellarGraph library. [See all other demos](../README.md).
# 

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/node-classification/gcn-node-classification.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/node-classification/gcn-node-classification.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

# [The StellarGraph library](https://github.com/stellargraph/stellargraph) supports many state-of-the-art machine learning (ML) algorithms on [graphs](https://en.wikipedia.org/wiki/Graph_%28discrete_mathematics%29). In this notebook, we'll be training a model to predict the class or label of a node, commonly known as node classification. We will also use the resulting model to compute vector embeddings for each node.
# 
# There's two necessary parts to be able to do this task:
# 
# - a graph: this notebook uses the Cora dataset from <https://linqs.soe.ucsc.edu/data>. The dataset consists of academic publications as the nodes and the citations between them as the links: if publication A cites publication B, then the graph has an edge from A to B. The nodes are classified into one of seven subjects, and our model will learn to predict this subject.
# - an algorithm: this notebook uses a Graph Convolution Network (GCN) [1]. The core of the GCN neural network model is a "graph convolution" layer. This layer is similar to a conventional dense layer, augmented by the graph adjacency matrix to use information about a node's connections. This algorithm is discussed in more detail in ["Knowing Your Neighbours: Machine Learning on Graphs"](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896).
# 
# The notebook walks through three sections:
# 
# 1. **Data preparation** using [Pandas](https://pandas.pydata.org) and [scikit-learn](https://scikit-learn.org/): loading the graph from CSV files, doing some basic introspection, and splitting it into train, test and validation splits for ML
# 2. **Creating the GCN layers** and data input using [StellarGraph](https://github.com/stellargraph/stellargraph)
# 3. **Training and evaluating** the model using [TensorFlow Keras](https://www.tensorflow.org/guide/keras), Pandas and scikit-learn
# 
# Notably, only section 2 needs StellarGraph: section 1 and section 3 are driven by the existing flexible functionality in common and popular data science libraries. Most of the algorithms supported by StellarGraph follow this pattern, where the custom StellarGraph functionality integrates smoothly with the conventional data science work-flow.
# 
# > StellarGraph supports other algorithms for doing [node classification](README.md), as well as many [other tasks](../README.md) such as [link prediction](../link-prediction/README.md), and [representation learning](../embeddings/README.md).
# 
# [1]: [Graph Convolutional Networks (GCN): Semi-Supervised Classification with Graph Convolutional Networks](https://github.com/tkipf/gcn). Thomas N. Kipf, Max Welling.
# International Conference on Learning Representations (ICLR), 2017

# The first step is to import the Python libraries that we'll need. We import `stellargraph` under the `sg` name for convenience, similar to `pandas` often being imported as `pd`.

# In[1]:


# In[2]:


# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

print(sg.utils.version_validation.__version__)
# In[3]:


import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt


# ## 1. Data Preparation
# 
# ### Loading the CORA network

# We can retrieve a `StellarGraph` graph object holding this Cora dataset using the `Cora` loader ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.datasets.Cora)) from the `datasets` submodule ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#module-stellargraph.datasets)). It also provides us with the ground-truth node subject classes. This function is implemented using Pandas, see [the "Loading data into StellarGraph from Pandas" notebook](../basics/loading-pandas.ipynb) for details.
# 
# (Note: Cora is a citation network, which is a directed graph, but, like most users of this graph, we ignore the edge direction and treat it as undirected.)

# (See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)

# In[4]:


train_vertex_file = "C:/SegmentGeneration/202112_c10_b_vertex.txt"
train_edge_file = "C:/SegmentGeneration/202112_c10_b_edges_data.txt"
train_graph_edges = pd.read_csv(train_edge_file, dtype='float')
graph_vertices = pd.read_csv(train_vertex_file,index_col="index", dtype='float')
G = sg.StellarGraph(graph_vertices.drop(['onRoad'], inplace=False, axis=1), train_graph_edges) #,'meanLatitude','meanLongitude','A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24'

train_node_types_dict = {}
for vertex in graph_vertices.index:
    train_node_types_dict[vertex] = float(graph_vertices.onRoad[vertex])
node_subjects = pd.Series(train_node_types_dict)


# In[5]:


train_graph_edges.dtypes


# The `info` method can help us verify that our loaded graph matches the description:

# In[6]:


print(G.info())


# We aim to train a graph-ML model that will predict the "subject" attribute on the nodes. These subjects are one of 7 categories, with some categories more common than others:

# In[7]:


node_subjects.value_counts().to_frame()


# ### Splitting the data

# For machine learning we want to take a subset of the nodes for training, and use the rest for validation and testing. We'll use scikit-learn's `train_test_split` function ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) to do this.
# 
# Here we're taking 140 node labels for training, 500 for validation, and the rest for testing.

# In[8]:


train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.8, test_size=None, stratify=node_subjects
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=0.5, test_size=None, stratify=test_subjects
)


# Note using stratified sampling gives the following counts:

# In[9]:


train_subjects.value_counts().to_frame()


# The training set has class imbalance that might need to be compensated, e.g., via using a weighted cross-entropy loss in model training, with class weights inversely proportional to class support. However, we will ignore the class imbalance in this example, for simplicity.

# ### Converting to numeric arrays

# For our categorical target, we will use one-hot vectors that will be compared against the model's soft-max output. To do this conversion we can use the `LabelBinarizer` transform ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)) from scikit-learn. Another option would be the `pandas.get_dummies` function ([docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)), but the scikit-learn transform allows us to do the inverse transform easily later in the notebook, to interpret the predictions.

# In[10]:


target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)


# In[11]:


train_subjects


# The CORA dataset contains attributes `w_x` that correspond to words found in that publication. If a word occurs more than once in a publication the relevant attribute will be set to one, otherwise it will be zero. These numeric attributes have been automatically included in the `StellarGraph` instance `G`, and so we do not have to do any further conversion.
# 
# <a href="Cora-features.png"><img src="Cora-features.png" alt="Each paper is analysed to see if it contains each of 1433 words" title="Click to zoom" width="600" /></a>

# ## 2. Creating the GCN layers
# 
# A machine learning model in StellarGraph consists of a pair of items:
# 
# - the layers themselves, such as graph convolution, [dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) and even [conventional dense layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
# - a data generator to convert the core graph structure and node features into a format that can be fed into the Keras model for training or prediction
# 
# GCN is a full-batch model and we're doing node classification here, which means the `FullBatchNodeGenerator` class ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.mapper.FullBatchNodeGenerator)) is the appropriate generator for our task. StellarGraph has many generators in order to support all [its many models and tasks](../README.md).
# 
# Specifying the `method='gcn'` argument to the `FullBatchNodeGenerator` means it will yield data appropriate for the GCN algorithm specifically, by using the [normalized graph Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix#Symmetric_normalized_Laplacian) to capture the graph structure.

# In[12]:


generator = FullBatchNodeGenerator(G, method="gcn")


# A generator just encodes the information required to produce the model inputs. Calling the `flow` method ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.mapper.FullBatchNodeGenerator.flow)) with a set of nodes and their true labels produces an object that can be used to train the model, on those nodes and labels that were specified. We created a training set above, so that's what we're going to use here.

# In[13]:


train_gen = generator.flow(train_subjects.index, train_targets)


# Now we can specify our machine learning model by building a stack of layers. We can use StellarGraph's `GCN` class ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.layer.GCN)), which packages up the creation of this stack of [graph convolution](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.layer.GraphConvolution) and [dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layers. We can specify a few parameters to control this:
# 
#  * `layer_sizes`: the number of hidden GCN layers and their sizes. In this case, two GCN layers with 16 units each.
#  * `activations`: the activation to apply to each GCN layer's output. In this case, [RelU](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\)) for both layers.
#  * `dropout`: the rate of dropout for the input of each GCN layer. In this case, 50%.

# In[14]:


gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
)


# To create a Keras model we now expose the input and output tensors of the GCN model for node prediction, via the `GCN.in_out_tensors` method:

# In[15]:


x_inp, x_out = gcn.in_out_tensors()

x_out


# The `x_out` value is a TensorFlow tensor that holds a 16-dimensional vector for the nodes requested when training or predicting. The actual predictions of each node's class/subject needs to be computed from this vector. StellarGraph is built using Keras functionality, so this can be done with a standard Keras functionality: an additional dense layer (with one unit per class) using a softmax activation. This activation function ensures that the final outputs for each input node will be a vector of "probabilities", where every value is between 0 and 1, and the whole vector sums to 1. The predicted class is the element with the highest value.

# In[16]:


predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)


# ## 3. Training and evaluating
# 
# ### Training the model

# Now let's create the actual [Keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) with the input tensors `x_inp` and output tensors being the predictions `predictions` from the final dense layer. Our task is a categorical prediction task, so a categorical cross-entropy loss function is appropriate. Even though we're doing graph ML with StellarGraph, we're still working with conventional Keras prediction values, so we can use [the loss function from Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy) directly.

# In[17]:


model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.binary_crossentropy,
    metrics=["acc"],
)


# As we're training the model, we'll want to also keep track of its generalisation performance on the validation set, which means creating another data generator, using our `FullBatchNodeGenerator` we created above.

# In[18]:


val_gen = generator.flow(val_subjects.index, val_targets)


# We can directly use the `EarlyStopping` functionality ([docs](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)) offered by Keras to stop training if the validation accuracy stops improving.

# In[19]:


from tensorflow.keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)


# In[20]:


[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]


# We've now set up our model layers, our training data, our validation data and even our training callbacks, so we can now train the model using the model's `fit` method ([docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)). Like most things in this section, this is all built into Keras.

# In[21]:


history = model.fit(
    train_gen,
    epochs=200,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    callbacks=[es_callback],
)


# Once we've trained the model, we can view the behaviour loss function and any other metrics using the `plot_history` function ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.utils.plot_history)). In this case, we can see the loss and accuracy on both the training and validation sets.

# In[ ]:


sg.utils.plot_history(history)


# As the final part of our evaluation, let's check the model against the test set. We again create the data required for this using the `flow` method on our `FullBatchNodeGenerator` from above, and can use the model's `evaluate` method ([docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate)) to compute the metric values for the trained model.
# 
# As expected, the model performs similarly on the validation set during training and on the test set here.

# In[ ]:


test_gen = generator.flow(test_subjects.index, test_targets)


# In[ ]:


test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


# ### Making predictions with the model

# Now let's get the predictions for all nodes. You're probably getting used to it by now, but we use our `FullBatchNodeGenerator` to create the input required and then use one of the model's methods: `predict` ([docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)). This time we _don't_ provide the labels to `flow`, and instead just the nodes, because we're trying to predict these classes without knowing them.

# In[ ]:


all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)


# These predictions will be the output of the softmax layer, so to get final categories we'll use the `inverse_transform` method of our target attribute specification to turn these values back to the original categories.
# 
# Note that for full-batch methods the batch size is 1 and the predictions have shape $(1, N_{nodes}, N_{classes})$ so we we remove the batch dimension to obtain predictions of shape $(N_{nodes}, N_{classes})$ using the NumPy `squeeze` method.

# In[ ]:


node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())


# Let's have a look at a few predictions after training the model:

# In[ ]:


df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
df.head(20)


# ### Node embeddings
# 
# In addition to just predicting the node class, it can be useful to get a more detailed picture of what information the model has learnt about the nodes and their neighbourhoods. In this case, this means an embedding of the node (also called a "representation") into a latent vector space that captures that information, and it comes in the form of either a look-up table mapping node to a vector of numbers, or a neural network that produces those vectors. For GCN, we're going to be using the second option, using the last graph convolution layer of the GCN model (called `x_out` above), before we applied the prediction layer.
# 
# We can visualise these embeddings as points on a plot, colored by their true subject labels. If the model has learned useful information about the nodes based on their class, we expect to see nice clusters of papers in the node embedding space, with papers of the same subject belonging to the same cluster.
# 
# To create a model that computes node embeddings, we use the same input tensors (`x_inp`) as the prediction model above, and just swap the output tensor to the GCN one (`x_out`) instead of the prediction layer. These tensors are connected to the same layers and weights that we trained when training the predictions above, and so we're only using this model to compute/"predict" the node embedding vectors. Similar to doing predictions for every node, we will compute embeddings for every node using the `all_gen` data.

# In[ ]:


embedding_model = Model(inputs=x_inp, outputs=x_out)


# In[ ]:


emb = embedding_model.predict(all_gen)
emb.shape


# The last GCN layer had output dimension 16, meaning each embedding consists of 16 numbers. Plotting this directly would require a 16 dimensional plot, which is hard for humans to visualise. Instead, we can first project these vectors down to just 2 numbers, making vectors of dimension 2 that can be plotted on a normal 2D scatter plot.
# 
# There are many tools for this [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) task, many of which are offered by scikit-learn. Two of the more common ones are [principal component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) (which is linear) and [t-distributed Stochastic Neighbor Embedding (t-SNE or TSNE)](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) (non-linear). t-SNE is slower but typically gives nicer results for plotting.

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

transform = TSNE  # or PCA


# Note that the embeddings from the GCN model have a batch dimension of 1 so we `squeeze` this to get a matrix of $N_{nodes} \times N_{emb}$.

# In[ ]:


X = emb.squeeze(0)
X.shape


# We've thus prepared our high-dimension embeddings and chosen our dimension-reduction transform, so we now compute the reduced vectors, as two columns of the new values.

# In[ ]:


trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
X_reduced.shape


# The `X_reduced` values contains a pair of numbers for each node, in the same order as the `node_subjects` Series of ground-truth labels (because that's how `all_gen` was created). This is enough to do a scatter plot of the nodes, with colors. We can let matplotlib compute the colors by mapping the subjects to integers 0, 1, ..., 6, using [Pandas's support for categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).
# 
# Qualitatively, the plot shows good clustering, where nodes of a single colour are mostly grouped together.

# In[ ]:


fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=node_subjects.astype("category").cat.codes,
    cmap="jet",
    alpha=0.7,
)
ax.set(
    aspect="equal",
    xlabel="$X_1$",
    ylabel="$X_2$",
    title=f"{transform.__name__} visualization of GCN embeddings for cora dataset",
)


# ## Conclusion
# 
# This notebook gave an example using the GCN algorithm to predict the class of nodes. Specifically, the subject of an academic paper in the Cora dataset. Our model used:
# 
# - the graph structure of the dataset, in the form of citation links between papers
# - the 1433-dimensional feature vectors associated with each paper
# 
# Once we trained a model for prediction, we could:
# 
# - predict the classes of nodes
# - use the model's weights to compute vector embeddings for nodes
# 
# This notebook ran through the following steps:
# 
# 1. prepared the data using common data science libraries
# 2. built a TensorFlow Keras model and data generator with [the StellarGraph library](https://github.com/stellargraph/stellargraph) 
# 3. trained and evaluated it using TensorFlow and other libraries
# 
# For problems with only small amounts of labelled data, model performance can be improved by semi-supervised training. See [the GCN + Deep Graph Infomax fine-tuning demo](gcn-deep-graph-infomax-fine-tuning-node-classification.ipynb) for more details on how to do this.
# 
# StellarGraph includes [other algorithms for node classification](README.md) and [algorithms and demos for other tasks](../README.md). Most can be applied with the same basic structure as this GCN demo.

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/node-classification/gcn-node-classification.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/node-classification/gcn-node-classification.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>
