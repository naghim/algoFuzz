#include "PFCM.h"
#include <iostream>
#include <limits> // For numeric_limits

PFCM::PFCM(int num_clusters, int max_iter, float m, int preprocess_iter, float p, float w_pos, float w_prob, float noise, bool fcplus1m)
	: BaseFCM(num_clusters, max_iter, m, noise), preprocess_iter(preprocess_iter), p(p), w_pos(w_pos), w_prob(w_prob), fcplus1m(fcplus1m)
{
	eta_values.resize(num_clusters);
	eta_values.setZero();
}

void PFCM::setParameters(const std::unordered_map<std::string, double> &params)
{
	BaseFCM::setParameters(params); // Call base class method
	if (params.find("preprocess_iter") != params.end())
	{
		preprocess_iter = static_cast<int>(params.at("preprocess_iter"));
	}
	if (params.find("p") != params.end())
	{
		p = static_cast<float>(params.at("p"));
	}
	if (params.find("w_pos") != params.end())
	{
		w_pos = static_cast<float>(params.at("w_pos"));
	}
	if (params.find("w_prob") != params.end())
	{
		w_prob = static_cast<float>(params.at("w_prob"));
	}
	if (params.find("fcplus1m") != params.end())
	{
		fcplus1m = static_cast<bool>(params.at("fcplus1m"));
	}
}

std::vector<std::string> PFCM::getParameterNames()
{
	auto base_params = BaseFCM::getParameterNames();
	base_params.push_back("preprocess_iter");
	base_params.push_back("p");
	base_params.push_back("w_pos");
	base_params.push_back("w_prob");
	base_params.push_back("fcplus1m");
	return base_params;
}

void PFCM::fit(const Eigen::MatrixXd &X_in)
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

	// Preprocessing with EtaFCM
	if (fcplus1m)
	{
		EtaPlus1FCM eta_fcm(num_clusters, preprocess_iter, m, 1.0f, noise); // kappa is 1.0 for EtaFCM preprocessing
		Eigen::MatrixXd centroids_copy = centroids;							// Create a copy of the centroids
		eta_fcm.setCentroids(centroids_copy);								// Use the copy for preprocessing
		eta_fcm.fit(X_in);													// Pass original X, as getXWithNoise is called internally
		eta_values = eta_fcm.getEta();
	}
	else
	{
		EtaFCM eta_fcm(num_clusters, preprocess_iter, m, 1.0f, noise); // kappa is 1.0 for EtaFCM preprocessing
		Eigen::MatrixXd centroids_copy = centroids;					   // Create a copy of the centroids
		eta_fcm.setCentroids(centroids_copy);						   // Use the copy for preprocessing
		eta_fcm.fit(X_in);											   // Pass original X, as getXWithNoise is called internally
		eta_values = eta_fcm.getEta();
	}

	float corrected_p = 1.0f / (p - 1.0f);
	float corrected_m = -2.0f / (m - 1.0f);

	for (int iter = 0; iter < max_iter; ++iter)
	{
		// Update u (probabilistic membership)
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
				float dist = (X.col(k) - centroids.col(i)).norm();
				if (dist < std::numeric_limits<float>::epsilon())
				{
					u(i, k) = std::numeric_limits<float>::infinity();
				}
				else
				{
					u(i, k) = std::pow(dist, corrected_m);
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
			for (int k = 0; k < n; ++k)
			{
				float dist_sq = (X.col(k) - centroids.col(i)).squaredNorm();
				float eta_val = eta_values(i);
				if (eta_val < std::numeric_limits<float>::epsilon())
				{
					// Handle case where eta_val is zero to avoid division by zero
					t(i, k) = 1.0f; // Or some other appropriate value
				}
				else
				{
					t(i, k) = 1.0f / (1.0f + std::pow((dist_sq * w_prob) / eta_val, corrected_p));
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
				float weight = w_pos * term_u + w_prob * term_t;
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
	// The Python implementation uses self.w_pos * u ** self.m + self.w_prob * t ** self.p for _member.
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