import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class BacktraderReport():
    def __init__(self):
        pass

    def run_report(self, strategy, outfilename):
        # TODO
        # This is a quick and easy way to draw PDF tables.
        # Need to find a better way that can take in a pandas dataframe.

        # Summary table
        summary_fig = plt.figure()
        summary_fig = plt.figure(figsize=(16, 5))

        summary_fig.suptitle('Trading Performance - Summary', fontsize=16)
        sharpe_ratio = strategy.analyzers.getbyname('sharperatio').get_analysis()['sharperatio']
        vwr = strategy.analyzers.getbyname('vwr').get_analysis()['vwr']
        sqn = strategy.analyzers.getbyname('sqn').get_analysis()
        sqn_text = str(sqn['sqn']) + ' (' + str(sqn['trades']) + ' trades)'

        cell_text = [['Sharpe Ratio', sharpe_ratio],
                     ['System Quality Number', sqn_text],
                     ['Variability-Weighted Return', vwr],
                     ['Scroll down for more...', '']]
        summary_table = plt.table(cellText=cell_text,
                                  loc='center')
        plt.axis('off')

        # Transactions & positions table
        row_labels = []
        row_colors = []
        cell_colors = []
        data_rows = []

        running_position = {}
        positions = {}
        symbols = []

        symbol_idx = 3
        price_idx = 1
        amount_idx = 0
        value_idx = 4

        transactions = strategy.analyzers.getbyname('transactions').get_analysis()
        odd_row = True

        for date, date_trans in transactions.items():
            row_color = "#dddddd" if odd_row else "#ffffff"
            cell_color = [row_color] * 5
            odd_row = not odd_row

            for trans in date_trans:
                row_labels.append(date)
                row_colors.append(row_color)
                cell_colors.append(cell_color)

                running_position[trans[symbol_idx]] = running_position.get(trans[symbol_idx], 0) + trans[amount_idx]

                data_rows.append([trans[symbol_idx],
                                  trans[price_idx],
                                  trans[amount_idx],
                                  running_position[trans[symbol_idx]],
                                  -round(trans[value_idx], 2)])

                if trans[symbol_idx] not in symbols:
                    symbols.append(trans[symbol_idx])

            if date not in positions:
                positions[date] = {}

            for symbol in symbols:
                positions[date][symbol] = running_position.get(symbol, 0)

        trans_fig = plt.figure(figsize=(16, 10))
        trans_fig.suptitle('UOB DeepCroc - Transactions', fontsize=16)
        trans_table = plt.table(cellText=data_rows,
                                rowLabels=row_labels,
                                rowColours=row_colors,
                                cellColours=cell_colors,
                                colLabels=['Symbol', 'Price', 'Change', 'Position', 'Value'],
                                loc='center')
        plt.axis('off')

        # Create position pivot table
        pos_row_labels = []
        pos_data_rows = []
        for pos_date, pos_list in positions.items():
            date_row = []
            for symbol in symbols:
                date_row.append(positions[pos_date].get(symbol, 0))

            pos_row_labels.append(pos_date)
            pos_data_rows.append(date_row)

        pos_fig = plt.figure(figsize=(16, 8))
        pos_fig.suptitle('UOB DeepCroc - Positions', fontsize=16)
        pos_table = plt.table(cellText=pos_data_rows,
                              rowLabels=pos_row_labels,
                              # rowColours=row_colors,
                              # cellColours=cell_colors,
                              colLabels=symbols,
                              loc='center')
        plt.axis('off')

        # Generate PDF
        pp = PdfPages(outfilename)
        pp.savefig(summary_fig)
        pp.savefig(pos_fig)
        pp.savefig(trans_fig)
        pp.close()