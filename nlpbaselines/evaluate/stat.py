from scipy.stats import ttest_ind, wilcoxon, shapiro


def p_value_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def perform_tests(list1, list2):
    shapiro_test_list1 = shapiro(list1)
    shapiro_test_list2 = shapiro(list2)

    test_result_ttest = ttest_ind(list1, list2)
    test_result_wilcoxon = wilcoxon(list1, list2)
    if shapiro_test_list1[1] > 0.05 and shapiro_test_list2[1] > 0.05:
        # If both lists pass the Shapiro test (p > 0.05), perform t-test
        Test_name = "T-test"
        print(
            f"Recommended test: {Test_name} because Shapiro test results for the two distributions are {shapiro_test_list1[1]} and {shapiro_test_list2[1]}. The p-value for the {Test_name} is {test_result_ttest[1]}"
        )
    else:
        # If either list fails the Shapiro test, perform Wilcoxon test
        Test_name = "Wilcoxon"
        print(
            f"Recommended test: {Test_name} because Shapiro test results for the two distributions are {shapiro_test_list1[1]} and {shapiro_test_list2[1]}. The p-value for the {Test_name} is {test_result_wilcoxon[1]}"
        )
    return {
        "T-test": test_result_ttest[1],
        "Wilcoxon": test_result_wilcoxon[1],
        "Test_name": Test_name,
    }


if __name__ == "__main__":
    print(perform_tests([1, 2, 3, 4, 5], [5, 6, 70, 8, 9]))
