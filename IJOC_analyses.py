import functools

def latexify_headers(*headers):
    headers = [h.replace('%', '\\%') for h in headers]
    headers = [h.replace(' ', '~') for h in headers]
    headers = ['\\mathrm{' + h + '}' for h in headers]
    return headers

def build_reference_row(data, **options):
    keys = options.get('keys')
    formats = options.get('formats')
    blanks = options.get('blanks')
    condition = options.get('condition')
    separator = options.get('separator')

    if condition == 'warm':
        data['warm']['warm_start_runs'] = data['meta']['warm_start_runs']

    row = '\n{} '
    values = [data['meta']['shape']]
    for i, fmt in enumerate(formats):
        row += '{sep} {} ' if blanks[i] else '{sep} ' + fmt + ' '
        values += ['--'] if blanks[i] else [data[condition][keys[i]]]
    return row.format(*values, sep=separator)

def build_row(data, **options):
    keys = options.get('keys')
    formats = options.get('formats')
    condition = options.get('condition')
    separator = options.get('separator')

    if condition == 'warm':
        data['warm']['warm_start_runs'] = data['meta']['warm_start_runs']

    row = '\n{}'
    values = [data['meta']['shape']]
    for i, fmt in enumerate(formats):
        row += '{sep} ' + fmt + ' '
        values.append(data[condition][keys[i]])
    return row.format(*values, sep=separator)


def table(full, *compressions, **options):
    CONDITION = options.pop('run_condition', 'cold')
    assert CONDITION in ('warm', 'cold'), 'warm start or cold start stats'

    LATEX = bool(options.pop('latex', False))
    MARKDOWN = not LATEX
    SEP = '&' if LATEX else '|'

    # body formatting
    row = functools.partial(
            build_row, condition=CONDITION, separator=SEP, **options)
    ref_row = functools.partial(
            build_reference_row, condition=CONDITION, separator=SEP, **options)


    # header formatting
    fields = options.get('fields')
    hdr = ' {sep}  {}' * len(fields)
    if LATEX:
        fields = latexify_headers(*fields)

    # build output
    # header
    output = '\n\\mathrm{dimension}' if LATEX else '\ndimension'
    output += hdr.format(*fields, sep=SEP)
    output += '\\\\' if LATEX else '\n-' + '|-' * len(fields)

    # full run row
    output +=  ref_row(full)
    output += '\\\\' if LATEX else ''

    # compressed run rows
    for summary in compressions:
        output += row(summary)
        output += '\\\\' if LATEX else ''

    if LATEX and options.pop('notebook', True):
        output = '\n$$\\begin{array}' + output
        output += '\n\\end{array} $$'
    return output

def collapse_table_cold(full, collapsed, **options):
    fields = ['time (s)']
    formats = ['{:0.3f}']
    blanks = [False]
    keys = ['primal_time']
    return table(
            full, collapsed,
            run_condition='cold',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)

def collapse_table_cold(full, collapsed, **options):
    fields = ['time (s)']
    formats = ['{:0.3f}']
    blanks = [False]
    keys = ['primal_time']
    return table(
            full, collapsed,
            run_condition='cold',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)

def collapse_table_warm(full, *compressions, **options):
    fields = ['mean time (s)', 'median time (s)']
    formats = ['{:0.3f}', '{:0.3f}']
    blanks = [False, False]
    keys = ['mean_primal_time', 'median_primal_time']
    return table(
            full, *compressions,
            run_condition='warm',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)

def vclu_table_warm(full, *compressions, **options):
    fields = [
            'mean time (s)', 'median time (s)',
            'mean subopt (%)', 'median subopt (%)',
            'mean true error (%)', 'median true error (%)',
            'runs']
    formats = [
            '{:0.3f}', '{:0.3f}',
            '{:0.1f}', '{:0.1f}',
            '{:0.1f}', '{:0.1f}',
            '{}']
    blanks = [
            False, False,
            True, True,
            True, True,
            False]
    keys = [
            'mean_primal_time', 'median_primal_time',
            'mean_suboptimality', 'median_suboptimality',
            'mean_true_error', 'median_true_error',
            'warm_start_runs']

    return table(
            full, *compressions,
            run_condition='warm',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)


def bclu_table_cold(full, *compressions, **options):
    fields = ['primal (s)', 'dual time (s)', 'subopt (%)', 'true error (%)']
    formats = ['{:0.3f}', '{:0.3f}', '{:0.1f}', '{:0.1f}']
    blanks = [False, True, True, True]
    keys = ['primal_time', 'dual_time', 'suboptimality', 'true_error']
    return table(
            full, *compressions,
            run_condition='cold',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)

def bclu_table_warm(full, *compressions, **options):
    fields = [
            'mean primal time (s)', 'median primal time (s)',
            'mean dual time (s)', 'median dual time (s)',
            'mean subopt (%)', 'median subopt (%)',
            'mean true error (%)', 'median true error (%)',
            'runs']
    formats = [
            '{:0.3f}', '{:0.3f}',
            '{:0.3f}', '{:0.3f}',
            '{:0.1f}', '{:0.1f}',
            '{:0.1f}', '{:0.1f}',
            '{}']
    blanks = [
            False, False,
            True, True,
            True, True,
            True, True,
            False]
    keys = [
            'mean_primal_time', 'median_primal_time',
            'mean_dual_time', 'median_dual_time',
            'mean_suboptimality', 'median_suboptimality',
            'mean_true_error', 'median_true_error',
            'warm_start_runs']

    return table(
            full, *compressions,
            run_condition='warm',
            fields=fields,
            formats=formats,
            blanks=blanks,
            keys=keys,
            **options)