from functions import *
import os

start_time = time.time()

# ......................................................
#
#   PARSE ARGUMENTS ----
#
# ......................................................
parser = argparse.ArgumentParser(description='Perform univariate statistical test such as T test or Zero inflated Wilcoxon.')
parser.add_argument('-i', '--input', type=str, help='Input count matrix path.')
parser.add_argument('-o', '--output_folder', type=str, help='Output folder path.')
parser.add_argument('-t', '--top_tags', type=float, default=200000, help='Top tags with best test statistics to keep.')
parser.add_argument('-c', '--chunk_size', type=int, default=10000, help='Size of each chunk in number of rows.')
parser.add_argument('-p', '--processes', type=int, default=mp.cpu_count(), help='Number of CPUs used/Default: number of CPUs available')
parser.add_argument('-d', '--condition_folder', type=str, help='Path to the condition folder.')
parser.add_argument('-n', '--normalize', nargs='?', const='default', type=str, help='Perform CPM normalization providing total k-mers file')
parser.add_argument('-f', '--norm_factor', type=int, default=1000000, help='Normalization factor')
parser.add_argument('--test_type', choices=['ttest', 'pitest', 'ziw','wilcoxon','variance', 'anova'], default='ttest', help='Test to perform and rank results.')
parser.add_argument('--covariates', type=str, default='no', help='Path to covariate CSV file or "no" to disable')

args = parser.parse_args()

# ......................................................
#
#   LOGGING ----
#
# ......................................................
script_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_directory, "chunk_processing.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

logging.info("Log file created and ready for new logging.")

# ......................................................
#
#   GUESS INPUT SEPARATOR ----
#
# ......................................................
with open(args.input) as f:
    head_lines = [next(f).rstrip() for x in range(2)]

input_separator = None
common_delimiters = [',', ';', '\t', ' ', '|', ':']
for d in common_delimiters:
    ref = head_lines[0].count(d)
    if ref > 0 and all(ref == head_lines[i].count(d) for i in range(1, 2)):
        input_separator = d
        break   # stop at first unambiguous delimiter

if input_separator is None:
    raise ValueError("Could not detect input file delimiter.")


# ......................................................
#
#   ESTIMATE TOTAL NUMBER OF TAGS (LINES) ----
#
# ......................................................
total_tags = estimate_total_lines(args.input)

if 0 < args.top_tags <= 1:
    args.top_tags = int(args.top_tags * total_tags)
else:
    args.top_tags = int(args.top_tags)
    
# ......................................................
#
#   READ COVARIATES DATA ----
#
# ......................................................
if args.covariates != "no":
    if args.covariates.endswith(".tsv"):
        covariates_df = pd.read_csv(args.covariates, sep="\t", index_col=0)
        logging.info("Covariates file read")
    else:
        raise ValueError("Covariates file must be a .tsv file.")
else:
    covariates_df = None

# ......................................................
#
#   BUILD NORMALIZATION DICT ONCE (if needed) ----
#   Previously this was re-read from disk on every chunk
#   inside normalize(). Now we do it once here.
# ......................................................
kmer_nb_dict = None
if args.normalize and args.normalize != 'default':
    design_kmer_nb = pd.read_csv(args.normalize, delimiter=' ')
    kmer_nb_dict = dict(zip(design_kmer_nb.iloc[:, 0], design_kmer_nb.iloc[:, 1]))
    logging.info("Normalization file read once.")

# ......................................................
#
#   READ HEADER ----
#
# ......................................................
with open(args.input, 'r') as file:
    header = file.readline().strip().split(input_separator)

pool = mp.Pool(processes=args.processes)

# ......................................................
#
#   EXECUTE STAT TEST ----
#
# ......................................................
top_tags_list = []
condition_files = [file for ext in ('*.tsv', '*.txt') for file in glob.glob(os.path.join(args.condition_folder, ext))]

for condition_file in condition_files:
    data_dict = create_data_dict(condition_file, args.test_type)

    func = functools.partial(
        work_for_parallel_processes,
        data_dict,
        cpm_normalization=kmer_nb_dict,
        header=header,
        test_type=args.test_type,
        covariates_df=covariates_df,
        norm_factor_c=args.norm_factor
    )

    chunk_iter = pd.read_csv(args.input, sep=input_separator, chunksize=args.chunk_size)
    result = pool.imap(func, chunk_iter)

    top_tags = []
    if args.test_type == 'pitest':
        for chunk_results in result:
            top_tags = keep_top_n(top_tags, chunk_results, args.top_tags, key_index=1, reverse=True)
        # Final sort: largest pi-value first
        top_tags.sort(key=lambda x: x[1], reverse=True)
        top_tags_list.append(top_tags)

    elif args.test_type in ('ttest', 'anova', 'wilcoxon'):
        for chunk_results in result:
            top_tags = keep_top_n(top_tags, chunk_results, args.top_tags, key_index=0, reverse=True)
        # Final sort: largest test statistic first
        top_tags.sort(key=lambda x: x[0], reverse=True)
        top_tags_list.append(top_tags)

    elif args.test_type == 'ziw':
        for chunk_results in result:
            top_tags = keep_top_n(top_tags, chunk_results, args.top_tags, key_index=0, reverse=True)
        # Final sort: largest test statistic first
        top_tags.sort(key=lambda x: x[0], reverse=True)
        top_tags_list.append(top_tags)

    elif args.test_type == 'variance':
        for chunk_results in result:
            top_tags = keep_top_n(top_tags, chunk_results, args.top_tags, key_index=0, reverse=True)
        # Final sort: largest variance first
        top_tags.sort(key=lambda x: x[0], reverse=True)
        top_tags_list.append(top_tags)

    logging.info("Condition file treated")

if args.test_type in ('ttest', 'anova', 'wilcoxon', 'ziw'):
    header.append('test_statistic')
    header.append('log2foldchange')
    header.append('p_value')
elif args.test_type == 'pitest':
    header.append('pivalue')
    header.append('log2foldchange')
elif args.test_type == 'variance':
    header.append('variance')
    header.append('cv')

# ......................................................
#
#   OUTPUT TOP K-MERS ----
#
# ......................................................
try:
    shutil.rmtree(args.output_folder)
except OSError:
    pass

os.makedirs(args.output_folder)

for condition_file, top_tags in zip(condition_files, top_tags_list):
    condition_name = os.path.splitext(os.path.basename(condition_file))[0]

    output_file = os.path.join(args.output_folder, f"{condition_name}.txt")
    with open(output_file, 'w') as file:
        file.write(' '.join(header) + '\n')

        if args.test_type in ('ttest', 'anova', 'wilcoxon'):
            for t_statistic, values, log2fold_change, p_value in top_tags:
                file.write(f"{values} {round(t_statistic, 2):g} {round(log2fold_change, 2):g} {p_value:.2g}\n")

        elif args.test_type == 'ziw':
            for test_statistic, values, log2fold_change, p_value in top_tags:
                file.write(f"{values} {round(test_statistic, 2):g} {round(log2fold_change, 2):g} {p_value:.2g}\n")

        elif args.test_type == 'pitest':
            for values, pivalue, log2fold_change in top_tags:
                file.write(f"{values} {pivalue} {log2fold_change}\n")
                
        elif args.test_type == 'variance':
            for variance_value, cv_value, values in top_tags:
                file.write(f"{values} {round(variance_value, 2):g} {round(cv_value, 2):g}\n")

end_time = time.time()
print("Execution time: {:.3f} s".format(end_time - start_time))
