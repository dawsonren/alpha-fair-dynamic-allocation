"""
Provide a generic interface to conduct generative random property tests.
"""


def random_property_test(generator, tester, instances=100):
    for _ in range(instances):
        instance = generator()
        tester(instance)
