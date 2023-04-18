"""Main non-linear function generator.
"""
import NFGen.funcs as nf
import NFGen.analysis as na
from NFGen.fitter import PolyK
# import NFGen.code_generator as cg

import sympy as sp
import numpy as np
import pickle
import time

DEBUG = False
save_all = False
profile_time = False
MAX_BREAKS = 10000


def sampled_poly_fit(a,
                     b,
                     k,
                     tol,
                     func,
                     f,
                     n,
                     ms=1e+3,
                     df=None,
                     zero_mask=None,
                     derivative_flag=True,
                     error_metric=("sampled", "relative_error"),
                     plain_method="OLS"):
    """Using the SampledPolyFit algorithm to find the valid candidate P_{(k, m)}.
    
    a: start of the input domain.
    b: end of the input domain.
    k: the target order.
    tol: user-defined tolerance.
    func: the target non-linear functions.
    f: the precision factor of target MPC fixed numbers.
    ms: default sampling limits.
    df: the pre-computed derivative function.
    zero_mask: for all value less than zero_mask, will be transformed to zero.
    derivative_flag: flag indicating whether the target func can calculate derivative.
    error_metric: the defined error analysis metric, now we only support sampled relative error and sampled absolute error.
    """

    def _merge_breaks(a,
                      b,
                      ms,
                      func,
                      k,
                      tol,
                      f,
                      n,
                      zero_mask=None,
                      derivative_flag=True,
                      error_metric=("sampled", "relative"),
                      plain_method="Cheby"):
        """Greedily merging all the mergeable successive pieces in P_{(k, m)}.
        """
        if derivative_flag:
            func_exec = nf.convert_function(func)
        else:
            func_exec = func
        # test merge or not.
        x, y = nf.adaptive_sampling(a, b, func_exec, f, ms, zero_mask)
        model = PolyK(k)
        model.fit(a, b, func, f, n, ms, derivative_flag, method=plain_method, zero_mask=zero_mask)

        if error_metric[0] == "sampled":
            y_pred = model.predict(x, n, f)
            each_error = nf.sampled_error(y, y_pred, f, zero_mask,
                                          error_metric[1])
            max_error = np.max(each_error)
            if DEBUG:
                argm = np.argmax(each_error)
                print("samples: ", len(y_pred))
                print(">>>>>> max_error: ", max_error)
                print("corresponding max error sample x = %.8g, y_true = %.8g, y_pred = %.8g"%(x[argm], y[argm], y_pred[argm]))

        elif error_metric[0] == "analytic":
            interval_coeff = model.coeff.squeeze() * model.scale.squeeze()
            max_error = nf.analytic_error(func,
                                          interval_coeff,
                                          x[0],
                                          x[-1],
                                          f,
                                          zero_mask,
                                          method=error_metric[1])
            if DEBUG:
                print(">> analytic max_error: ", max_error)

        if max_error < tol:
            return True, model.coeff, model.scale
        else:
            return False, None, None

    if DEBUG:
        print("In fit: start - %.8f, end - %.8f" % (a, b))

    decimal = nf.calculate_decimal(f)

    # FAC algorithm
    result = []
    coeff_list = []
    scale_list = []
    stack = [(a, b)]
    poss_breaks = 1

    # Function conversion
    if derivative_flag and df is None:
        df = nf.find_derivative(func)
        func_exec = nf.convert_function(func)
    else:
        func_exec = func

    while (stack):
        if poss_breaks >= MAX_BREAKS:
            # Too long pieces.
            raise Exception("Breaks exceed the limit.")

        start, end = stack.pop()
        if DEBUG:
            print("Start - %.18f | End - %.18f" % (start, end))

        if (np.double(start).round(decimal) == np.double(end).round(decimal)):
            continue

        # Fit polynomial
        x, y = nf.adaptive_sampling(start, end, func_exec, f, ms, zero_mask=zero_mask)
        
        if(len(x) == 1):
            print(">>>> y: ", y)
            model.coeff = np.concatenate([y, [0]*(k)])[:, np.newaxis]
            model.scale = np.ones(model.coeff.shape)
        else:
            model = PolyK(k)
            model.fit(start, end, func, f, n, ms=ms, derivative_flag=derivative_flag, method=plain_method, zero_mask=zero_mask)
        
        split_flag = True
        if error_metric[0] == "sampled":
            y_pred = model.predict(x, n, f)
            each_error = nf.sampled_error(y, y_pred, f, zero_mask,
                                          error_metric[1])
            max_error = np.max(each_error)
            split_flag = (max_error >= tol)
            
            if DEBUG:
                argm = np.argmax(each_error)
                print("samples: ", len(y_pred))
                print(">>>>>> max_error: ", max_error)
                print("corresponding max error sample x = %.8g, y_true = %.8g, y_pred = %.8g"%(x[argm], y[argm], y_pred[argm]))
                print(">> split_flag here: ", split_flag)

        elif error_metric[0] == "analytic":
            interval_coeff = model.coeff.squeeze() * model.scale.squeeze()
            max_error = nf.analytic_error(func,
                                          interval_coeff,
                                          start,
                                          end,
                                          f,
                                          zero_mask,
                                          method=error_metric[1])
            split_flag = (max_error > tol)
            
            if DEBUG:
                print(">> analytic max_error: ", max_error)
                print(">> analytic split_flag: ", split_flag)

        # Succeed accuracy test or not
        if split_flag and (len(x) > (k + 1)):
            poss_breaks += 1
            stack.append(((start + end) / 2, end))
            stack.append((start, (start + end) / 2))
        else:
            coeff_list.append(model.coeff)
            scale_list.append(model.scale)
            result.append([start, end])

    # Greedy merge.
    rp = 0
    while (rp < len(result) - 1):
        new_breaks = [result[rp][0], result[rp + 1][1]]
        flag, new_coeff, new_scale = _merge_breaks(new_breaks[0], new_breaks[1], ms, func, k, tol, f, n,
                                                   zero_mask, derivative_flag,
                                                   error_metric, plain_method=plain_method)
        if flag:
            for _ in range(2):
                coeff_list.pop(rp)
                scale_list.pop(rp)
                result.pop(rp)

            coeff_list.insert(rp, new_coeff)
            scale_list.insert(rp, new_scale)
            result.insert(rp, new_breaks)
        else:
            rp += 1

    return coeff_list, result, scale_list


def generate_nonlinear_config(config_dict):
    """Main function to generate the non-linear functions' config.

    Args:
        config_dict (dict): Dict stores the config information for target non-linear functions.
    """

    # Necessary keys check
    if "function" not in config_dict.keys():
        print("Please indicate the target function F in `function`.")
    if "range" not in config_dict.keys():
        print("Please indicate the input domain [a, b] in `range`.")
    if "tol" not in config_dict.keys():
        print("Please indicate the tolerance \epsilon in `tol`.")
    if "n" not in config_dict.keys():
        print("Please indicate the number length in `n`.")
    if "f" not in config_dict.keys():
        print("Please indicate the resolution in `f`.")
    if "profiler" not in config_dict.keys():
        print("Please indicate profiler model(some .pkl file) in `profiler`.")
    if "code_templet" not in config_dict.keys():
        print("Please indicate templet in `code_templet`.")
        
    method = "Cheby"

    func = config_dict['function']
    a, b = config_dict['range']
    tol = config_dict['tol']
    n = config_dict['n']
    f = config_dict['f']
    decimal = nf.calculate_decimal(f)
    profiler_file = config_dict['profiler']
    pf = open(profiler_file, 'rb')
    mk_profiler = pickle.load(pf)
    pf.close()
    # code_templet = config_dict['code_templet']

    # Other hints, and default values.
    if "k_max" not in config_dict.keys():
        k_range = range(3, 10)
    else:
        k_range = range(3, config_dict['k_max'])

    if "zero_mask" not in config_dict.keys():
        zero_mask = 1e-8
    else:
        zero_mask = config_dict['zero_mask']

    if "config_file" not in config_dict.keys():
        config_file = './config_file.py'
    else:
        config_file = config_dict['config_file']

    if "ms" in config_dict.keys():
        ms = config_dict['ms']
    else:
        ms = 1000

    if "not_check" in config_dict.keys(
    ):  # whether check F_s or not, default to False.
        not_check_flag = config_dict['not_check']
    else:
        not_check_flag = False

    if "nick_name" in config_dict.keys():  # nick function name.
        save_name = config_dict["nick_name"]
    else:
        save_name = str(func).split(" ")[1]

    # if "code_language" in config_dict.keys(
    # ):  # difference generation methods, specific for python or cpp.
    #     language = config_dict["code_language"]
    # else:
    #     language = "python"

    if "derivative_flag" in config_dict.keys(
    ):  # flag indicating whether the target function is derivative or not, if derivative, must be written as SymPy function.
        derivative_flag = config_dict["derivative_flag"]
    else:
        derivative_flag = True

    if "default_values" in config_dict.keys(
    ):  # default values exceeds the input domain.
        default_flag = True
        left_default = config_dict["default_values"][0]
        right_default = config_dict["default_values"][1]
        less_break = a - 999
        larger_break = b + 999
    else:
        default_flag = False

    error_metric = ("sampled", "relative_error")
    if "error_metric" in config_dict.keys():
        error_metric = config_dict["error_metric"]

    if "time_dict" in config_dict.keys():
        basic_time_dict = config_dict['time_dict']
    else:
        not_check_flag = False

    candidate_list = []

    if profile_time:
        time_start = time.time()

    # Generate the candidate P_{(k, m)}.
    for k in k_range:
        if DEBUG:
            coeff_list, breaks, scale_list = sampled_poly_fit(
                a,
                b,
                k,
                tol,
                func,
                f,
                n,
                ms=ms,
                df=None,
                zero_mask=zero_mask,
                derivative_flag=derivative_flag,
                error_metric=error_metric,
                plain_method=method)
        else:
            try:
                coeff_list, breaks, scale_list = sampled_poly_fit(
                    a,
                    b,
                    k,
                    tol,
                    func,
                    f,
                    n,
                    ms=ms,
                    df=None,
                    zero_mask=zero_mask,
                    derivative_flag=derivative_flag,
                    error_metric=error_metric,
                    plain_method=method)
            except Exception as e:
                print(e.args)
                print("failed current k = %d" % k)
                continue
        breaks, coeffA, scale_list = nf.result_orgnize(breaks, coeff_list,
                                                       scale_list)
        breaks = np.array(breaks)
        scale_list = np.array(scale_list)

        if (len(breaks) == 2):
            coeffA = coeffA[np.newaxis, :]
            scale_list = scale_list[np.newaxis, :]
        candidate_list.append((len(breaks), k, coeffA, breaks, scale_list))

    if profile_time:
        print("Profile all the candidates: " )
        for ele_tuple in candidate_list:
            print("(k, m) = ", (len(ele_tuple[2][0]), len(ele_tuple[3])))
        time_candidate = time.time()
        print("Candidate construction: ", time_candidate - time_start)

    if save_all:
        i = 0
        for ele_tuple in candidate_list:
            breaks, coeffA, scale_list = ele_tuple[3], ele_tuple[2], ele_tuple[
                4]
            coeffA = coeffA.round(decimal)  # ronuded coefficient.
            scale_list = scale_list.round(
                decimal)  # directly save the reciprocal of scales.
            kc, m = ele_tuple[1], ele_tuple[0]

            # Generate graph
            if derivative_flag:
                function = nf.convert_function(func)
            else:
                function = func

            if "test_graph" in config_dict.keys():
                x = np.linspace(a, b, 100000)

                if error_metric[0] == "sampled":
                    y_true = function(x)
                    y_pred = nf.piece_prediction(x[:, np.newaxis], coeffA, breaks,
                                                scale_list, f, n)
                    breaks_label = nf.piece_prediction(
                        np.array(breaks)[:, np.newaxis], coeffA, breaks,
                        scale_list, f, n)
                    na.sampled_error_analysis(graph_folder,
                                            save_name,
                                            kc,
                                            x,
                                            y_true,
                                            y_pred,
                                            breaks,
                                            breaks_label,
                                            f,
                                            zero_mask=zero_mask,
                                            method=error_metric[1])
                elif error_metric[0] == "analytic":
                    na.analytic_error_analysis(graph_folder,
                                            save_name,
                                            kc,
                                            x,
                                            breaks,
                                            coeffA,
                                            scale_list,
                                            func,
                                            f,
                                            zero_mask=zero_mask,
                                            method=error_metric[1])

            if "default_values" in config_dict.keys():
                breaks = np.insert(breaks, 0, less_break)
                scaler_default = np.array([1.0] *
                                          len(scale_list[0]))[np.newaxis]
                scale_list = np.insert(scale_list, 0, scaler_default, axis=0)
                scale_list = np.append(scale_list, scaler_default, axis=0)

                coeff_left, coeff_right = np.zeros(
                    (1, len(coeffA[0]))), np.zeros((1, len(coeffA[0])))
                coeff_left[0, 0] = left_default
                coeff_right[0, 0] = right_default
                coeffA = np.insert(coeffA, 0, coeff_left, axis=0)
                coeffA = np.append(coeffA, coeff_right, axis=0)

            # save config.
            # na.generate_config(config_file, save_name + "_%d" % (i), coeffA,
            #                    breaks, scale_list)
            kmconfig = {
                'breaks': breaks.tolist(),
                'coeffA': coeffA.tolist(),
                'scaler': scale_list.tolist()
            }
            print("candidate KM: ",
                (len(kmconfig['coeffA'][0]), len(kmconfig['breaks'])))
            # cg.code_generate(kmconfig,
            #                 mk_profiler,
            #                 func,
            #                 basic_time_dict,
            #                 code_templet,
            #                 cg.basic_building_blocks,
            #                 config_file,
            #                 nick_name=save_name+"_%d"%(i),
            #                 not_check=not_check_flag,
            #                 code_language=language)

            i += 1

        return

    # Select the best coefficient and save.
    mk_list = np.array([[candidate_list[i][0], candidate_list[i][1]]
                        for i in range(len(candidate_list))])
    best_index = np.argmin(mk_profiler.predict(mk_list))
    best = candidate_list[best_index]
    if profile_time:
        time_selection = time.time()
        print("Best Selection: ", time_selection - time_candidate)

    breaks, coeffA, scale_list = best[3], best[2], best[4]
    coeffA = coeffA.round(decimal)  # ronuded coefficient.
    scale_list = scale_list.round(
        decimal)  # directly save the reciprocal of scales.

    # Generate graph for demonstration
    if "test_graph" in config_dict.keys():
        graph_folder = config_dict['test_graph']

        if derivative_flag:
            function = nf.convert_function(func)
        else:
            function = func

        x = np.linspace(a, b, 10000)

        if error_metric[0] == "sampled":
            y_true = function(x)
            y_pred = nf.piece_prediction(x[:, np.newaxis], coeffA, breaks,
                                         scale_list, f, n)
            breaks_label = nf.piece_prediction(
                np.array(breaks)[:, np.newaxis], coeffA, breaks, scale_list, f, n)
            na.sampled_error_analysis(graph_folder,
                                      save_name,
                                      best[1],
                                      x,
                                      y_true,
                                      y_pred,
                                      breaks,
                                      breaks_label,
                                      f,
                                      zero_mask=zero_mask,
                                      method=error_metric[1])
        elif error_metric[0] == "analytic":
            na.analytic_error_analysis(graph_folder,
                                       save_name,
                                       best[1],
                                       x,
                                       breaks,
                                       coeffA,
                                       scale_list,
                                       func,
                                       f,
                                       zero_mask=zero_mask,
                                       method=error_metric[1])

    if default_flag:
        breaks = np.insert(breaks, 0, less_break)
        breaks = np.append(breaks, larger_break)
        scaler_default = np.array([1.0] * len(scale_list[0]))[np.newaxis]
        scale_list = np.insert(scale_list, 0, scaler_default, axis=0)
        scale_list = np.append(scale_list, scaler_default, axis=0)

        coeff_left, coeff_right = np.zeros((1, len(coeffA[0]))), np.zeros(
            (1, len(coeffA[0])))
        coeff_left[0, 0] = left_default
        coeff_right[0, 0] = right_default
        coeffA = np.insert(coeffA, 0, coeff_left, axis=0)
        coeffA = np.append(coeffA, coeff_right, axis=0)

    # Code generation
    kmconfig = {
        'breaks': breaks.tolist(),
        'coeffA': coeffA.tolist(),
        'scaler': scale_list.tolist()
    }

    import json
    polysFile = "./%s.json"%(save_name)
    with open(polysFile, "w") as f:
        json.dump(kmconfig, f, indent=4)

    print(">>>>> FINAL KM: ",
          (len(kmconfig['coeffA'][0]), len(kmconfig['breaks'])))
    # cg.code_generate(kmconfig,
    #                  mk_profiler,
    #                  func,
    #                  basic_time_dict,
    #                  code_templet,
    #                  cg.basic_building_blocks,
    #                  config_file,
    #                  nick_name=save_name,
    #                  not_check=not_check_flag,
    #                  code_language=language)

    # if profile_time:
    #     time_end = time.time()
    #     time_code_generation = time_end - time_selection
    #     print("Time code generation: ", time_code_generation)
    #     print("Time all: ", time_end - time_start)
    #     return time_end - time_start


    