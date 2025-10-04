#include "FCPlus1M.h"

FCPlus1M::FCPlus1M(int num_clusters, int max_iter, float m, float kappa, float eta, float noise)
	: FCM(num_clusters, max_iter, m, kappa, noise), eta(eta)
{
}

void FCPlus1M::setParameters(const std::unordered_map<std::string, double> &params)
{
	FCM::setParameters(params); // Call base class method
	if (params.find("eta") != params.end())
	{
		eta = static_cast<float>(params.at("eta"));
	}
}

std::vector<std::string> FCPlus1M::getParameterNames()
{
	auto base_params = FCM::getParameterNames();
	base_params.push_back("eta");
	return base_params;
}

float FCPlus1M::calculate_initial_sum()
{
	float corrected_m = -2.0f / (m - 1.0f);
	return std::pow(eta, corrected_m);
}