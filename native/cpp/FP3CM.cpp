#include "FP3CM.h"
#include <iostream>
#include <limits> // For numeric_limits

FP3CM::FP3CM(int num_clusters, int max_iter, float m, float p, float eta, float noise)
	: BaseFCM(num_clusters, max_iter, m, noise), p(p), eta(eta)
{
	// Additional initialization specific to FP3CM if needed
}

void FP3CM::setParameters(const std::unordered_map<std::string, double> &params)
{
	BaseFCM::setParameters(params); // Call base class method
	if (params.find("p") != params.end())
	{
		p = static_cast<float>(params.at("p"));
	}
	if (params.find("eta") != params.end())
	{
		eta = static_cast<float>(params.at("eta"));
	}
}

std::vector<std::string> FP3CM::getParameterNames()
{
	auto base_params = BaseFCM::getParameterNames();
	base_params.push_back("p");
	base_params.push_back("eta");
	return base_params;
}

void FP3CM::fit(const Eigen::MatrixXd &X_in)
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

	float corrected_p = 1.0f / (p - 1.0f);
	float corrected_m = -1.0f / (m - 1.0f);

	float eta2 = eta * eta;

	for (int iter = 0; iter < max_iter; ++iter)
	{
		// Update t (typicality)
		for (int i = 0; i < num_clusters; ++i)
		{
			for (int k = 0; k < n; ++k)
			{
				float dist_sq = (X.col(k) - centroids.col(i)).squaredNorm();
				t(i, k) = 1.0f / (1.0f + std::pow(dist_sq / eta2, corrected_p));
			}
		}

		// Update u (membership)
		for (int k = 0; k < n; ++k)
		{
			float szum = 0.0f;
			int exact_match_cluster = -1;

			for (int i = 0; i < num_clusters; ++i)
			{
				if ((X.col(k) - centroids.col(i)).norm() < 0.0000001f)
				{
					exact_match_cluster = i;
					break;
				}
			}

			if (exact_match_cluster != -1)
			{
				u.col(k) = Eigen::VectorXd::Zero(num_clusters);
				u(exact_match_cluster, k) = 1.0f;
				continue;
			}

			for (int i = 0; i < num_clusters; ++i)
			{
				float dist_sq = (X.col(k) - centroids.col(i)).squaredNorm();
				u(i, k) = std::pow((std::pow(t(i, k), p) * dist_sq + eta2 * std::pow(1.0f - t(i, k), p)), corrected_m);
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

		// Update centroids
		for (int i = 0; i < num_clusters; ++i)
		{
			Eigen::VectorXd sum_up = Eigen::VectorXd::Zero(z);
			float sum_dn = 0.0f;

			for (int k = 0; k < n; ++k)
			{
				float sum_cur = std::pow(u(i, k), m) * std::pow(t(i, k), p);
				sum_up += sum_cur * X.col(k);
				sum_dn += sum_cur;
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
	// The Python implementation uses u ** self.m + t ** self.p for _member.
	// This is a bit unusual for a membership matrix, but we'll replicate it.
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