# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""

from collections import Counter

import matplotlib.cm as cm
import numpy as np
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler, OneHotEncoder, LabelEncoder


class Transformer(BaseEstimator, TransformerMixin):

	def __init__(self):
		self.random_state = 777
		self.is_test = False

	def fit(self, X, y=None, **kwargs):

		if isinstance(X, tuple):
			self.X = X[0]
			self.y = X[1]
		else:
			self.X = X
			self.y = y

		self.n_cols = self.X.shape[1]

	def transform(self, X, y=None, **kwargs):

		if isinstance(X, tuple):
			X = X[0]

		self.X = X

	@classmethod
	def pipeline_wrapper(cls, train, test, pipeline):

		print("Building pipeline ...")
		pipeline_ = Pipeline(pipeline)

		train_tr = pipeline_.fit_transform(train)
		if isinstance(train_tr, tuple):
			train_tr = train_tr[0]

		test_tr = test.copy()

		for pipe, transformer in pipeline:
			pipeline_.named_steps[pipe].is_test = True
			test_tr = pipeline_.named_steps[pipe].transform(test_tr)

		transformed = np.concatenate([train_tr, test_tr], axis=0)
		transformed = [transformed[:, i:i + 1] for i in range(transformed.shape[1])]

		print("Number of instances: {}\nNumber of features: {}".format(transformed[0].shape[0], len(transformed)))

		return transformed


class DataFrameSelector(Transformer):  # Pandas to numpy selector

	def __init__(self, attribute_names, y=None):
		"""
		:param attribute_names: list of column names of the pandas dataframe
		"""
		super().__init__()

		self.attribute_names = attribute_names
		self.y = y
		self.length = len(attribute_names)

		print("Number of attributes selected: {}".format(str(self.length)))

	def fit(self, X, y=None, **kwargs):
		# super().fit(X, y)
		return self

	def transform(self, X,  y=None, **kwargs):
		super().transform(X, y)

		if not self.is_test:
			if self.y is not None:
				return self.X[self.attribute_names].values.reshape(-1, self.length), self.X[self.y].values.reshape(-1, 1)
			else:
				return self.X[self.attribute_names].values.reshape(-1, self.length)
		else:
			return self.X[self.attribute_names].values.reshape(-1, self.length)


class MissingEncoder(Transformer):

	def __init__(self, is_missing=np.nan):
		"""
		:param is_missing: list of placehoder/ placehoder of missing value for each columns;
		"""
		super().__init__()
		self.missing = is_missing

	def fit(self, X, y=None):

		super().fit(X, y)
		return self

	def transform(self, X, y=None, **kwargs):

		super().transform(X, y)

		if not isinstance(self.missing, list):
			self.missing = [self.missing] * self.n_cols
		self.missing = np.array(self.missing).reshape(-1, self.n_cols, 1)
		X_transformed = np.where(self.X.reshape(-1, self.n_cols, 1) == self.missing, 1, 0).reshape(-1, self.n_cols)

		if not self.is_test:
			return X_transformed, self.y
		else:
			return X_transformed


class CategoricalMeanTransformer(Transformer):

	"""
	Is is_test necessary to make train test distinction?
	"""

	def __init__(self, noise=0.1, is_test=False):

		super().__init__()
		self.is_test = is_test
		self.noise = noise if not None else 0

	def fit(self, X, y=None):

		super().fit(X, y)
		self.groupmean = {}
		self.threshold = self.X.shape[0] * 0.005
		self.loo = np.zeros_like(self.X, dtype=float)
		self.noise_array = np.zeros_like(self.X, dtype=float)
		self.global_mean = self.y.mean()
		self.X = self.X.astype('int64')

		for col in range(self.X.shape[1]):
			print("Processing feature {}".format(col))
			self.X[:, col] = self.X[:, col].astype(int)
			denominator = Counter(self.X[:, col])
			numerator = Counter(self.X[:, col:col + 1][np.where(self.y == 1)])

			self.groupmean[col] = {k: numerator[k] / v if v > self.threshold else self.global_mean for k, v in denominator.items()}
			self.loo[:, col: col + 1] = np.array([(numerator[self.X[i, col]] - self.y[i]) / (denominator[self.X[i, col]] - 1) if denominator[self.X[i, col]] != 1 else self.global_mean for i in range(self.X.shape[0])]).reshape(-1,1)  # leave-one-out
			np.random.seed(self.random_state + col)
			self.noise_array[:, col:col + 1] = np.random.normal(1, self.noise, self.X.shape[0]).reshape(-1, 1) if self.noise else np.ones(self.X.shape[0], 1)  # random noise

		# self.loo[np.isnan(self.loo)] = self.global_mean # New group level in test set will be replaced by global mean
		return self

	def transform(self, X, y=None, **kwargs):

		super().transform(X, y)

		if self.is_test:
			print("Start categorical mean transforming test data")
			X_transformed = np.zeros_like(self.X, dtype=float)

			for col in range(self.X.shape[1]):
				groupmean = self.groupmean[col]
				X_transformed[:, col:col + 1] = np.array([groupmean.get(self.X[i, col], self.global_mean) for i in range(self.X.shape[0])]).reshape(-1,1)

			return X_transformed

		elif self.y is not None:
			print("Start categorical mean transforming training data in leave-one-out mode")
			X_transformed = self.loo * self.noise_array
			return X_transformed, self.y

		elif self.y is None:
			print("Try fitting the data first")


class kMeansTransformer(Transformer):
	def __init__(self, n_cluster=None, gamma=None):

		super().__init__()
		self.n_clusters = n_cluster
		self.gamma = 1.0 if gamma is None else gamma

	def fit(self, X, y=None, compute_score=False):

		super().fit(X, y)
		if self.n_clusters:
			self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_jobs=-2)
			print("Computing label")
			self.cluster_labels = self.clusterer.fit_predict(self.X)

			if compute_score:
				self.silhouette_avg = silhouette_score(self.X, self.cluster_labels)
				print("For n_clusters =", self.n_clusters, "The average silhouette_score is :", self.silhouette_avg)

			if self.y is not None:
				print("Normalised mutual information score: {:.4}".format(normalized_mutual_info_score(self.y.ravel(), self.cluster_labels)))

		return self

	def transform(self, X, y=None, return_cluster=False, **kwargs):

		super().transform(X, y=None)
		print("Start kMeans transforming ... ")

		cluster_rbf = np.exp(- self.gamma * self.clusterer.transform(self.X))

		if not self.is_test and self.y is not None:
			return cluster_rbf, self.y
		else:
			return cluster_rbf

	def plot(self, X = None):

		if X is not None:
			self.X

		sample_silhouette_values = silhouette_samples(self.X, self.cluster_labels)

		y_lower = 10

		fig, ax1 = plt.subplots(1, 1)
		for i in range(self.n_clusters):
			ith_cluster_silhouette_values = sample_silhouette_values[self.cluster_labels == i]
			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.spectral(float(i) / self.n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax1.axvline(x=self.silhouette_avg, color="red", linestyle="--")


class tSVDTransformer(Transformer):
	"""
	Apply truncated SVD transformation to numeric and categorical numeric and categorical dummy.
	Preprocessing usually is required (e.g., standardisation).
	"""

	def __init__(self, n_components=None, min_components=5, max_components=100):

		super().__init__()
		self.n_components = n_components
		self.min_components = min_components
		self.max_components = max_components

	def fit(self, X, y=None):

		super().fit(X, y)
		if not self.n_components:
			self.tsvd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
			self.tsvd.fit(self.X)
			self.explained_ratio = self.tsvd.explained_variance_ratio_.sum()
			print(self.tsvd.explained_variance_ratio_)
		else:
			for i in range(max(self.min_components, self.n_components), min(self.max_components, sekf.X.shape[1]), 5):
				self.tsvd = TruncatedSVD(n_components=i, random_state=self.random_state)
				self.tsvd.fit(self.X)
				self.explained_ratio = self.tsvd.explained_variance_ratio_.sum()
				print("{} components variance explained {:.2%}".format(i, self.explained_ratio))
				if self.explained_ratio > 0.9:
					break

		return self

	def transform(self, X, y=None, **kwargs):

		super().transform(X, y)
		print("Start tSVD transformation ... ")

		if not self.is_test and self.y is not None:
			return self.tsvd.transform(self.X), self.y
		else:
			return self.tsvd.transform(self.X)


class CategoricalCountTransformer(Transformer):

	def __init__(self):
		super().__init__()
		self.counts = {}

	def fit(self, X, y=None):
		"""
		:param X: Numpy array with shape(n_sample, n_feature)
		:return:
		"""
		super().fit(X, y)
		self.n_cols = self.X.shape[1]

		for col in range(self.n_cols):
			self.counts[col] = Counter(self.X[:, col])
		return self

	def transform(self, X, y=None, **kwargs):

		super().transform(X, y)
		print("Start categorical count transforming ... ")

		X_transformed = np.zeros_like(self.X)

		for col in range(self.n_cols):
			X_transformed[:, col] = np.array(
				[self.counts[col][val] if val in self.counts[col].keys() else 0 for val in self.X[:, col]])

		if not self.is_test and self.y is not None:
			return X_transformed, self.y
		else:
			return X_transformed


class PercentileClipper(Transformer):

	def __init__(self, low_percentile=1, up_percentile=99):
		super().__init__()

		self.low_percentile = low_percentile
		self.up_percentile = up_percentile

	def fit(self, X, y=None):
		super().fit(self.X, y)
		self.floor = []
		self.ceiling = []
		self.n_cols = X.shape[1]

		self.floor = np.percentile(X, self.low_percentile, axis=0)
		self.ceiling = np.percentile(X, self.up_percentile, axis=0)

		return self

	def transform(self, x, y=None, **kwargs):

		super().transform(X, y)
		print("Start percentile clipping ... ")

		if not self.is_test and self.y is not None:
			return np.clip(self.X, self.floor, self.ceiling), self.y
		else:
			return np.clip(self.X, self.floor, self.ceiling)


class NumericBoxCoxTransformer(Transformer):

	def __init__(self):
		super().__init__()
		self.lmbda = []

	def fit(self, X, y=None):
		super().fit(X, y)
		self.n_cols = self.X.shape[1]

		for col in range(self.n_cols):
			self.lmbda.append(boxcox(self.X[:, col])[1])

		return self

	def transform(self, X, y=None, **kwargs):

		super().transform(X, y)
		print("Start boxcox transformation ... ")

		X_transformed = np.zeros_like(self.X)
		for col in range(self.n_cols):
			X_transformed[:, col] = boxcox(self.X[:, col], self.lmbda[col])

		if not self.is_test and self.y is not None:
			return X_transformed, self.y
		else:
			return X_transformed


class StandardTransformer(Transformer):

	def __init__(self):

		super().__init__()
		self.scaler = StandardScaler()

	def fit(self, X, y=None):

		super().fit(X, y)
		self.scaler.fit(self.X)

		return self

	def transform(self, X, y=None):

		print("Start standardising data ...")
		super().transform(X, y)

		if not self.is_test and self.y is not None:
			return self.scaler.transform(self.X), self.y
		else:
			return self.scaler.transform(self.X)


class MinMaxTransformer(Transformer):

	def __init__(self):

		super().__init__()
		self.scaler = MinMaxScaler()

	def fit(self, X, y=None):

		super().fit(X, y)
		self.scaler.fit(self.X)

		return self

	def transform(self, X, y=None):

		print("Start normalising data ...")
		super().transform(X, y)

		if not self.is_test and self.y is not None:
			return self.scaler.transform(self.X), self.y
		else:
			return self.scaler.transform(self.X)


class MissingImputer(Transformer):

	def __init__(self, missing_values, strategy):

		super().__init__()
		self.imputer = Imputer(missing_values=missing_values, strategy=strategy)

	def fit(self, X, y=None):

		super().fit(X, y)
		self.imputer.fit(self.X)

		return self

	def transform(self, X, y=None):

		print("Start filling missing value ...")
		super().transform(X, y)

		if not self.is_test and self.y is not None:
			return self.imputer.transform(self.X), self.y
		else:
			return self.imputer.transform(self.X)


class Replacer(Transformer):

	def __init__(self, target_list, replace_list):

		super().__init__()

		self.target_list = [target_list] if not isinstance(target_list, list) else target_list
		self.replace_list = [replace_list] if not isinstance(replace_list, list) else replace_list
		assert len(self.target_list) == len(self.replace_list)


	def fit(self, X, y=None):

		super().fit(X, y)

		return self

	def transform(self, X, y=None, **kwargs):

		print("Start replacing value ...")
		super().transform(X, y)

		transformed_X = self.X

		for i in range(0, self.X.shape[1]):
			transformed_X[:, i] = np.array([ self.replace_list[i] if val == self.target_list[i] else val for val in self.X[:,i]])

		if not self.is_test and self.y is not None:
			return transformed_X, self.y
		else:
			return transformed_X


class OneHotTransformer(Transformer):

	def __init__(self,num_values):

		super().__init__()
		self.ohe = OneHotEncoder(n_values=num_values, sparse=False, handle_unknown='ignore')

	def fit(self, X, y=None):

		super().fit(X, y)
		self.ohe.fit(self.X)

		return self

	def transform(self, X, y=None):

		print("Start one-hot transforming ...")
		super().transform(X, y)

		if not self.is_test and self.y is not None:
			return self.ohe.transform(self.X), self.y
		else:
			return self.ohe.transform(self.X)


class TransformerUnion(Transformer):

	def __init__(self, pipelines):
		super().__init__()
		self.pipelines = pipelines

	def fit(self, X):

		for pipeline in self.pipelines:
			pipeline.fit(X)

		return self

	def transform(self, X):

		transformed = []

		for i, pipeline in enumerate(self.pipelines):
			print("Processing pipeline {}".format(i))
			transformed.append(pipeline.transform(X))

		transformed = np.column_stack(transformed)
		print("Number of instances: {}\nNumber of features: {}".format(transformed.shape[0], transformed.shape[1]))

		return transformed


if __name__ == "__main__":

	attribute_names = []

	pipes = [
		('selector', DataFrameSelector(attribute_names)),
		('std_scaler', StandardScaler())
	]
