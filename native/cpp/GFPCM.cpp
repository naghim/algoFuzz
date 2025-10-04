#include "GFPCM.h"
#include <iostream>
#include <limits> // For numeric_limits
#include <random> // For random_device, mt19937, uniform_int_distribution

GFPCM::GFPCM(int num_clusters, int max_iter, float m, float p, float w_prob, float noise)
	: BaseFCM(num_clusters, max_iter, m, noise), p(p), w_prob(w_prob)
{
	// Additional initialization specific to GFPCM if needed
}

void GFPCM::setParameters(const std::unordered_map<std::string, double> &params)
{
	BaseFCM::setParameters(params); // Call base class method
	if (params.find("p") != params.end())
	{
		p = static_cast<float>(params.at("p"));
	}
	if (params.find("w_prob") != params.end())
	{
		w_prob = static_cast<float>(params.at("w_prob"));
	}
}

std::vector<std::string> GFPCM::getParameterNames()
{
	auto base_params = BaseFCM::getParameterNames();
	base_params.push_back("p");
	base_params.push_back("w_prob");
	return base_params;
}

void GFPCM::fit(const Eigen::MatrixXd &X_in)
{
	if (!centroids_set)
	{
		throw std::runtime_error("Centroids must be set before calling fit.");
	}

	Eigen::MatrixXd X = getXWithNoise(X_in);

	int z = X.rows(); // Number of features (dimensions)
	int n = X.cols(); // Number of samples (data points)

	member.resize(num_clusters, n);
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(num_clusters, n);
	Eigen::MatrixXd t = Eigen::MatrixXd::Zero(num_clusters, n);

	float deriv_m = -2.0f / (m - 1.0f);
	float deriv_p = -2.0f / (p - 1.0f);

	for (int iter = 0; iter < max_iter; ++iter)
	{
		// Update u (probabilistic membership)
		for (int k = 0; k < n; ++k)
		{
			float szum = 0.0f;
			for (int i = 0; i < num_clusters; ++i)
			{
				float dist = (X.col(k) - centroids.col(i)).norm();
				if (dist < std::numeric_limits<float>::epsilon())
				{
					u(i, k) = std::numeric_limits<float>::infinity();
				}
				else
				{
					u(i, k) = std::pow(dist, deriv_m);
				}
				szum += u(i, k);
			}
			for (int i = 0; i < num_clusters; ++i)
			{
				if (szum > std::numeric_limits<float>::epsilon())
				{
					u(i, k) /= szum;
				}
				else
				{
					u(i, k) = 0.0f;
				}
			}
		}

		// Update t (possibilistic membership)
		for (int i = 0; i < num_clusters; ++i)
		{
			float szum = 0.0f;
			for (int k = 0; k < n; ++k)
			{
				float dist = (X.col(k) - centroids.col(i)).norm();
				if (dist < std::numeric_limits<float>::epsilon())
				{
					t(i, k) = std::numeric_limits<float>::infinity();
				}
				else
				{
					t(i, k) = std::pow(dist, deriv_p);
				}
				szum += t(i, k);
			}
			for (int k = 0; k < n; ++k)
			{
				if (szum > std::numeric_limits<float>::epsilon())
				{
					t(i, k) /= szum;
				}
				else
				{
					t(i, k) = 0.0f;
				}
			}
		}

		// Update centroids
		for (int i = 0; i < num_clusters; ++i)
		{
			Eigen::VectorXd sum_up = Eigen::VectorXd::Zero(z);
			float sum_dn = 0.0f;

			for (int k = 0; k < n; ++k)
			{
				float term_u = std::pow(u(i, k), m);
				float term_t = std::pow(t(i, k), p);
				float weight = term_u + w_prob * term_t;
				sum_up += weight * X.col(k);
				sum_dn += weight;
			}

			if (sum_dn > std::numeric_limits<float>::epsilon())
			{
				centroids.col(i) = sum_up / sum_dn;
			}
			else
			{
				// Keep previous centroid value if sum_dn is zero
			}
		}
	}

	// After iterations, set member matrix and trained flag
	// The Python implementation uses u ** self.m + self.w_prob * t ** self.p for _member.
	// For predicted labels, we'll use the 'u' matrix as it represents fuzzy membership.
	member = u; // We'll use 'u' as the primary membership for label prediction

	// Calculate predicted labels
	predictedLabels.clear();
	predictedLabels.reserve(n);
	for (int k = 0; k < n; ++k)
	{
		Eigen::MatrixXf::Index max_row;
		u.col(k).maxCoeff(&max_row);
		predictedLabels.push_back(static_cast<int>(max_row));
	}

	trained = true;
}
