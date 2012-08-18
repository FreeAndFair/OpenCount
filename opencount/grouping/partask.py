import sys, os, pdb, multiprocessing, random, math

"""
A utility script/tool that handles performing an embarassingly
parallelizable task.
I've done the same pattern enough times to tire of doing it again...
might as well abstract the pattern.
"""

def do_partask(fn, jobs, _args=None, blocking=True, combfn=None, init=None):
    """ The developer-facing main function. do_partask will split up
    'jobs' into N equally-sized chunks C_i, and apply 'fn' to each
    C_i in parallel, in addition to (optionally) providing additional
    arguments (from '_args', a list of things).
    Input:
        fn: A function that must be of the following signature:
              fn(list, arg_0, ..., arg_N)
            where each arg_i is optional, and provided by '_args'
        list jobs: A list of tasks, to be split up by do_partask.
        list _args: 
        bool blocking: 
        fn combfn: A combining function that must take two arguments
            combfn(T results, T* subresults) -> (T results*, int k)
        T init: Used with 'combfn', specifies the initial starting 
                value.
    Output:
        The return value of calling 'fn' on all things in 'jobs', in a
        flat-list.
    TODO: non-blocking functionality is not implemented.
    """
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    p = multiprocessing.Process(target=spawn_jobs, args=(queue, fn, jobs, _args))
    p.start()

    num_jobs = len(jobs)
    if combfn == None:
        results = []
        while True:
            subresults = queue.get()
            if isinstance(subresults, POOL_CLOSED):
                return results
            results.extend(subresults)
        return results
    else:
        results = init
        while True:
            subresults = queue.get()
            if isinstance(subresults, POOL_CLOSED):
                return results
            results = combfn(results, subresults)
        return results

class POOL_CLOSED:
    """
    A dummy class to use to signal that the pool is finished
    processing.
    """
    def __init__(self):
        pass

_POOL_CLOSED = POOL_CLOSED()

def spawn_jobs(queue, fn, jobs, _args=None):
    def handle_result(result):
        queue.put(result)
    pool = multiprocessing.Pool()
    n_procs = multiprocessing.cpu_count()
    for i, job in enumerate(divy_list(jobs, n_procs)):
        print "Process {0} got {1} tasks.".format(i, len(job))
        if _args == None:
            pool.apply_async(fn, args=(job,), callback=handle_result)
        else:
            pool.apply_async(fn, args=(job, _args), callback=handle_result)            
    pool.close()
    pool.join()
    queue.put(POOL_CLOSED())
    
def divy_list(lst, k):
    """ Divides input list into 'k' equally-sized chunks.
    Input:
        list lst: A list of stuff.
        int k: An integer.
    Output:
        K tuples.
    """
    if len(lst) <= k:
        return [lst]
    chunksize = math.floor(len(lst) / float(k))
    i = 0
    chunks = []
    curchunk = []
    while i < len(lst):
        if i != 0 and ((i % chunksize) == 0):
            chunks.append(curchunk)
            curchunk = []
        curchunk.append(lst[i])
        i += 1
    if curchunk:
        chunks.append(curchunk)
    return chunks
    
"""
Test scripts. Refer to this usage to get an idea of how to use this
utility script.
"""

def square_nums(nums):
    return [x * x for x in nums]

def test0():
    nums = [random.random() for _ in range(1)]
    nums_squared = do_partask(square_nums, nums)
    print "Input nums:"
    print nums
    print "Output nums:"
    print nums_squared
    
def test1():
    nums = [random.random() for _ in range(10000)]
    # Fascinating -- 'square' must be defined as a function with 'def',
    # you can't just do:
    # square = lambda x: x * x
    nums_squared = do_partask(square_nums, nums)
    print "Input nums:"
    print nums[0:10]
    print "Output nums:"
    print nums_squared[0:10]

def eval_quadratic(nums, (a, b, c)):
    """ 
    Evalutes 'x' with the following quadratic:
        a*x**2 + b*x + c
    """
    return [a*(x**2) + b*x + c for x in nums]

def test2():
    nums = [random.random() for _ in range(10000)]
    result = do_partask(eval_quadratic, nums, _args=[4, 3, 2])
    print "Input nums:"
    print nums[0:4]
    print "Output nums:"
    print result[0:4]

def count_occurs(strs):
    res = {}
    for str in strs:
        if str in res:
            res[str] += 1
        else:
            res[str] = 1
    return res

def count_combfn(d, subd):
    """ Combines two dictionaries. """
    final_d = dict(d.items())
    for k, v in subd.iteritems():
        if k in final_d:
            final_d[k] += v
        else:
            final_d[k] = v
    return final_d

def test3():
    strs = ('a', 'a', 'a', 'a', 'a', 'a',
            'b', 'b', 'b', 'b',
            'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
            'a', 'a',
            'z', 'z', 'z')
    counts = do_partask(count_occurs, strs, combfn=count_combfn, init={})
    print counts

def run_tests():
    print "==== Running tests..."
    print "== 1.) Square 1 number."
    test0()

    print "== 2.) Square 10,000 numbers."
    test1()

    print "== 3.) Eval a quadratic-expression of 10,000 numbers."
    test2()

    print "== 4.) Counting occurrences of strings."
    test3()

    print "==== Completed tests."

if __name__ == '__main__':
    run_tests()
