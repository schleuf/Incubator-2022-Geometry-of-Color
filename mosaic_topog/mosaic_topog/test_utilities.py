import pytest
import mosaic_topog.utilities as util


def test_numSim_1():
    """
    check that numSim throws an exception for an empty input
    """
    process = 'a'
    num_sim = []
    sim_to_gen = ['a', 'b', 'c']
    with pytest.raises(Exception):
        util.numSim(process, num_sim, sim_to_gen)

    
def test_numSim_2():
    """
    check that numSim throws an exception for a num_sim list of 
    length > 1 but < length of sim_to_gen list
    """
    process = 'a'
    num_sim = [1, 2]
    sim_to_gen = ['a', 'b', 'c']
    with pytest.raises(Exception):
        util.numSim(process, num_sim, sim_to_gen)


def test_numSim_3():
    """
    check that numSim throws an exception for a num_sim list of 
    length > 1 and length of sim_to_gen list
    """
    process = 'a'
    num_sim = [1, 2, 3, 4, 5]
    sim_to_gen = ['a', 'b', 'c']
    with pytest.raises(Exception):
        util.numSim(process, num_sim, sim_to_gen)

       
def test_numSim_4():
    """
    check that numSim returns the right value when a list of length 1 is
    input for num_sim
    """
    process = 'a'
    num_sim = [1]
    sim_to_gen = ['a', 'b', 'c']
    test = util.numSim(process, num_sim, sim_to_gen)
    assert (test == 1)
    process = 'b'
    test = util.numSim(process, num_sim, sim_to_gen)
    assert (test == 1)
    process = 'c'
    test = util.numSim(process, num_sim, sim_to_gen)
    assert (test == 1)

    
def test_numSim_5():
    """
    check that numSim returns the right value when a list of values
    corresponding to each sim is given
    """
    num_sim = [1, 2, 3]
    sim_to_gen = ['a', 'b', 'c']
    test = util.numSim('a', num_sim, sim_to_gen)
    assert (test == 1)
    test = util.numSim('b', num_sim, sim_to_gen)
    assert (test == 2)
    test = util.numSim('c', num_sim, sim_to_gen)
    assert (test == 3)
    