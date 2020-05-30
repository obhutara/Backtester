import pandas as pd
import os
import sys
import numpy as np
import datetime

# D:\STOCK TICK DATA\APR-2019\GFDLNFO_TICK_01042019\GFDLNFO_TICK_01042019\Futures\-I

def get_data(ticker_name = None,start_date = None,end_date = None,future_month=None):
    '''
    Add Getting Data Logic Here
    :param ticker_name
    :param start_date:
    :param end_date:
    :return:
    '''
    final_df = pd.DataFrame()
    start_date = datetime.datetime.strptime(start_date,'%d/%m/%Y')
    end_date = datetime.datetime.strptime(end_date,'%d/%m/%Y')
    days_in_between = (end_date - start_date).days
    all_dates = [start_date+datetime.timedelta(days=x) for x in range(0,days_in_between)]
    print(all_dates)
    for x in all_dates:
        try:
            month = x.strftime('%b').upper()
            year = x.strftime('%Y')
            date_format_new = x.strftime('%d%m%Y')
            final_path = f"D:\STOCKTICKDATA\{month}-{year}\GFDLNFO_TICK_{date_format_new}\GFDLNFO_TICK_{date_format_new}\Futures\-{future_month}\{ticker_name}-{future_month}.NFO.csv"
            print(final_path)
            df = pd.read_csv(final_path)
            df = df[df['LTQ'] != 0]
            df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df['Turnover'] = (df['LTP'] * df['LTQ']).cumsum().tail(1).values[0]
            # print(df['Turnover'].tail(1))
            final_df = final_df.append(df, ignore_index=True)
        except FileNotFoundError:
            print(f"Not a Trading Day {date_format_new}")
        except:
            continue
    print(final_df,final_df.columns)
    final_df.sort_values(by=['Date_Time'], ascending=True, inplace=True)
    return final_df


def do_backtest(df = None,ticker_name = None,start_price=None,transaction_type = None,entry_diff = None,exit_diff=None,max_open_lot=None,max_loss=None,lot_size=None,lot_each_step=None):
    '''
    ticker_name = 'MANAA,start_price=106,transaction_type = 'BUY,lots=10,entry_diff = 1,exit_diff=1,max_open_lot=10,max_loss=None,lot_size=6000,lot_each_step=1
    :param ticker_name: Name of the Ticker
    :param transaction_type: Buy/Sell
    :param lots: Number of Lots
    :param entry_diff: Entry Difference
    :param exit_diff: Exit Difference
    :param max_open_lot: Maximum number of Open Lots
    :param max_loss: maximum loss to take after which all the positions are closed
    :param lot_size: Number of Shares/Lot
    :return: Trade sheet
    '''
    try:
        # df = get_data()
        # file_names = set(['MANAPPURAM-I.NFO (1).csv', 'MANAPPURAM-I.NFO (2) (2).csv'
        #               ,'MANAPPURAM-I.NFO.csv'])
        #
        # df = get_data_files(file_names = file_names)
        trades_df = pd.DataFrame() ## Completed Trades Come Here
        ledger  = pd.DataFrame()
        trade_waiting_to_execute = []
        trade_executed = []
        if transaction_type == 'BUY':
            multiplier = 1
        elif transaction_type == 'SELL':
            multiplier = -1
        for x in range(0, int(max_open_lot/lot_each_step)):
            trade_waiting_to_execute.append({'Multiplier':multiplier,'Entry_Execution_Time':None,'Exit_Execution_Time':None,'Entry_Execution_Price':None,'Entry_Price':start_price - multiplier*x*entry_diff , 'Transaction_Type' : transaction_type, 'Exit_Price':start_price- multiplier*x*entry_diff+ multiplier*exit_diff,'Quantity':lot_each_step*lot_size })
        print(trade_waiting_to_execute)

        last_data = None
        temp_ledger = dict()
        temp_ledger['Timestamp'] = None
        temp_ledger['Closing BUY Qty (Cumul)'] = 0
        temp_ledger['Closing BUY AVG Price'] = 0
        temp_ledger['Closing SELL Qty (Cumul)'] = 0
        temp_ledger['Closing SELL AVG Price'] = 0
        temp_ledger['OI'] = 0
        temp_ledger['Realized Profit'] = 0
        len_df = len(df)
        for index,data in df.iterrows():
            print((index+1)*100/len_df," % Completed")
            last_data = data
            temp_ledger['Timestamp'] = data['Date_Time']
            temp_ledger['Opening BUY Qty (Cumul)'] = temp_ledger['Closing BUY Qty (Cumul)']
            temp_ledger['Opening BUY AVG Price'] = temp_ledger['Closing BUY AVG Price']
            temp_ledger['Opening SELL Qty (Cumul)'] = temp_ledger['Closing SELL Qty (Cumul)']
            temp_ledger['Opening SELL AVG Price'] = temp_ledger['Closing SELL AVG Price']
            temp_ledger['Sell Qty (Ank)'] = 0
            temp_ledger['Buy Qty (Ank)'] = 0
            LTP = data['LTP']
            LTQ = data['LTQ']
            temp_ledger['Start OI'] = temp_ledger['OI']
            temp_ledger['LTQ'] = data['LTQ']
            temp_ledger['Turnover'] = data['Turnover']


            temp_add_to_executed = []
            temp_add_to_waiting_execute = []
            # print("INDEX",index)
            for x in range(0,len(trade_waiting_to_execute)):
                this_trade = trade_waiting_to_execute[x]
                if this_trade['Transaction_Type'] == 'BUY':

                    if LTP <= this_trade['Entry_Price'] and LTQ >= this_trade['Quantity']:
                        if index > 0:
                            print("Entry happened")
                            last_ltp = df.loc[index-1,'LTP']
                            trade_waiting_to_execute[x]['Entry_Execution_Price'] = this_trade['Entry_Price']
                            this_trade['Entry_Execution_Price'] = this_trade['Entry_Price']
                            trade_waiting_to_execute[x]['Entry_Execution_Time'] = data['Date_Time']
                            this_trade['Entry_Execution_Time'] = data['Date_Time']
                            LTQ -=  this_trade['Quantity']
                            temp_add_to_executed.append(this_trade)
                            temp_ledger['Closing BUY AVG Price'] = (temp_ledger['Closing BUY AVG Price'] * temp_ledger['Closing BUY Qty (Cumul)']+ this_trade['Quantity'] * this_trade['Entry_Execution_Price'])/(this_trade['Quantity']+temp_ledger['Closing BUY Qty (Cumul)'])
                            temp_ledger['Closing BUY Qty (Cumul)'] += this_trade['Quantity']
                            temp_ledger['Buy Qty (Ank)'] += this_trade['Quantity']
                            continue

                        else:
                            trade_waiting_to_execute[x]['Entry_Execution_Price'] = LTP
                            this_trade['Entry_Execution_Price'] = LTP
                            LTQ -= this_trade['Quantity']
                            trade_waiting_to_execute[x]['Entry_Execution_Time'] = data['Date_Time']
                            this_trade['Entry_Execution_Time'] = data['Date_Time']
                            temp_ledger['Closing BUY AVG Price'] = (temp_ledger['Closing BUY AVG Price'] * temp_ledger['Closing BUY Qty (Cumul)']+ this_trade['Quantity'] * this_trade['Entry_Execution_Price'])/(this_trade['Quantity']+temp_ledger['Closing BUY Qty (Cumul)'])
                            temp_ledger['Closing BUY Qty (Cumul)'] += this_trade['Quantity']
                            temp_ledger['Buy Qty (Ank)'] += this_trade['Quantity']
                            temp_add_to_executed.append(this_trade)


                elif this_trade['Transaction_Type'] == 'SELL':
                    if LTP >= this_trade['Entry_Price'] and LTQ >= this_trade['Quantity']:
                        print("Entry happened")
                        if index > 0:
                            trade_waiting_to_execute[x]['Entry_Execution_Time'] = data['Date_Time']
                            this_trade['Entry_Execution_Time'] = data['Date_Time']
                            trade_waiting_to_execute[x]['Entry_Execution_Price'] = this_trade['Entry_Price']
                            LTQ -= this_trade['Quantity']
                            this_trade['Entry_Execution_Price'] = this_trade['Entry_Price']
                            temp_ledger['Closing SELL AVG Price'] = (temp_ledger['Closing SELL AVG Price'] * temp_ledger['Closing SELL Qty (Cumul)']+ this_trade['Quantity'] * this_trade['Entry_Execution_Price'])/(this_trade['Quantity']+temp_ledger['Closing SELL Qty (Cumul)'])
                            temp_ledger['Closing SELL Qty (Cumul)'] += this_trade['Quantity']
                            temp_ledger['Sell Qty (Ank)'] += this_trade['Quantity']
                            temp_ledger['Turnover'] = data['Turnover']
                            temp_add_to_executed.append(this_trade)

                            continue
                        else:
                            trade_waiting_to_execute[x]['Entry_Execution_Time'] = data['Date_Time']
                            this_trade['Entry_Execution_Time'] = data['Date_Time']
                            trade_waiting_to_execute[x]['Entry_Execution_Price'] = LTP
                            LTQ -= this_trade['Quantity']
                            this_trade['Entry_Execution_Price'] = LTP
                            temp_ledger['Closing SELL AVG Price'] = (temp_ledger['Closing SELL AVG Price'] * temp_ledger['Closing SELL Qty (Cumul)']+ this_trade['Quantity'] * this_trade['Entry_Execution_Price'])/(this_trade['Quantity']+temp_ledger['Closing SELL Qty (Cumul)'])
                            temp_ledger['Closing SELL Qty (Cumul)'] += this_trade['Quantity']
                            temp_ledger['Sell Qty (Ank)'] += this_trade['Quantity']
                            temp_ledger['Turnover'] = data['Turnover']
                            temp_add_to_executed.append(this_trade)

            for x in range(0,len(trade_executed)):
                this_trade = trade_executed[x]
                if this_trade['Transaction_Type'] == 'BUY':
                    ### We need to exit this Buy side if target price is reached
                    if LTP >= this_trade['Exit_Price'] and LTQ >= this_trade['Quantity']:
                        this_trade['Exit_Execution_Time'] = data['Date_Time']
                        temp_ledger['Closing SELL AVG Price'] = (temp_ledger['Closing SELL AVG Price'] *temp_ledger['Closing SELL Qty (Cumul)']+this_trade['Exit_Price']*this_trade['Quantity'])/(temp_ledger['Closing SELL Qty (Cumul)']+this_trade['Quantity'])
                        temp_ledger['Closing SELL Qty (Cumul)'] += this_trade['Quantity']
                        temp_ledger['Realized Profit'] = temp_ledger['Realized Profit'] + this_trade['Quantity']*(this_trade['Exit_Price'] - this_trade['Entry_Execution_Price'])
                        temp_ledger['Sell Qty (Ank)'] += this_trade['Quantity']
                        temp_ledger['Turnover'] = data['Turnover']
                        temp_add_to_waiting_execute.append(this_trade)
                elif this_trade['Transaction_Type'] == 'SELL':
                    ### We need to exit this Buy side if target price is reached
                    if LTP <= this_trade['Exit_Price'] and LTQ >= this_trade['Quantity']:
                        this_trade['Exit_Execution_Time'] = data['Date_Time']
                        temp_ledger['Closing BUY AVG Price'] = (temp_ledger['Closing BUY AVG Price'] *temp_ledger['Closing BUY Qty (Cumul)']+this_trade['Exit_Price']*this_trade['Quantity'])/(temp_ledger['Closing BUY Qty (Cumul)']+this_trade['Quantity'])
                        temp_ledger['Closing BUY Qty (Cumul)'] += this_trade['Quantity']
                        temp_ledger['Realized Profit'] = temp_ledger['Realized Profit'] + this_trade['Quantity']*(this_trade['Exit_Price'] - this_trade['Entry_Execution_Price'])*-1
                        temp_ledger['Buy Qty (Ank)'] += this_trade['Quantity']
                        temp_ledger['Turnover'] = data['Turnover']
                        temp_add_to_waiting_execute.append(this_trade)

            temp_ledger['LTP'] = LTP
            temp_ledger['OI'] = data['OpenInterest']
            temp_ledger['Turnover'] = data['Turnover']
            ledger = ledger.append(temp_ledger,ignore_index=True)

            for x in range(0,len(temp_add_to_executed)):
                # These were Executed so need to be removed from waiting to execute and have to be added to
                this_trade = temp_add_to_executed[x]
                trade_waiting_to_execute.remove(this_trade)
                trade_executed.append(this_trade)

            temp_add_to_executed = []

            for x in range(0,len(temp_add_to_waiting_execute)):
                this_trade =  temp_add_to_waiting_execute[x]
                trades_df = trades_df.append(this_trade,ignore_index=True) ## The Whole trade was completed so Added to Trades DF
                trade_executed.remove(this_trade)
                trade_waiting_to_execute.append(this_trade)

            temp_add_to_waiting_execute = []

            # print(len(trade_waiting_to_execute),len(trade_executed))
            # assert(len(trade_waiting_to_execute)+len(trade_executed)==max_open_lot,"ERROR")

        for x in range(0, len(trade_executed)):
            this_trade = trade_executed[x]
            this_trade['Message'] = "Position Open"
            this_trade['Exit_Price'] = last_data['LTP']
            this_trade['Exit_Execution_Time'] = None
            trades_df = trades_df.append(this_trade,ignore_index=True)

        ledger['Realized Profit'].ffill(inplace=True)
        ledger['Unrealized Profit'] = np.where(ledger['Closing BUY Qty (Cumul)']*multiplier  > ledger['Closing SELL Qty (Cumul)']*multiplier,(ledger['Closing BUY Qty (Cumul)']-ledger['Closing SELL Qty (Cumul)'])*(ledger['LTP'] - ledger['Closing BUY AVG Price']),None)
        ledger['Unrealized Profit'] = np.where(ledger['Closing SELL Qty (Cumul)']  >= ledger['Closing BUY Qty (Cumul)'],(-1*ledger['Closing BUY Qty (Cumul)']+ledger['Closing SELL Qty (Cumul)'])*(ledger['Closing SELL AVG Price'] - ledger['LTP']),ledger['Unrealized Profit'])
        # ledger['Turnover'] = data['Turnover']

        trades_df['Profit'] = (trades_df['Exit_Price'] - trades_df['Entry_Execution_Price'])*trades_df['Multiplier']*trades_df['Quantity']
        trades_df['Exit_Execution_Time'] = pd.to_datetime(trades_df['Exit_Execution_Time'])
        ledger = ledger[['Timestamp','Opening BUY AVG Price', 'Opening BUY Qty (Cumul)','Buy Qty (Ank)', 'Opening SELL AVG Price',
             'Opening SELL Qty (Cumul)','Sell Qty (Ank)', 'Closing BUY Qty (Cumul)', 'Closing BUY AVG Price',
              'LTP', 'Closing SELL Qty (Cumul)', 'Closing SELL AVG Price',
              'Turnover',
             'Realized Profit', 'Unrealized Profit', 'OI','LTQ']]

        return trades_df,ledger
    except:
        print(sys.exc_info())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def generate_report(ledger = None,transaction_type = 'BUY'):
    if transaction_type == 'BUY':
        multiplier = 1
    elif transaction_type == 'SELL':
        multiplier = -1

    ledger['Date'] = ledger['Timestamp'].dt.date
    g = ledger.groupby(by=['Date'])
    heads = g.head(1)
    tails = g.tail(1)[['Date','Closing BUY AVG Price', 'Closing BUY Qty (Cumul)', 'LTP', 'Closing SELL AVG Price',
                 'Closing SELL Qty (Cumul)'
                 , 'Realized Profit', 'Unrealized Profit','OI','Turnover']]
    main_df = pd.DataFrame()
    main_df = main_df.append(heads[['Date','Opening BUY AVG Price', 'Opening BUY Qty (Cumul)', 'Opening SELL AVG Price',
                 'Opening SELL Qty (Cumul)']],ignore_index=True)
    main_df = pd.merge(main_df,tails,how='left',on=['Date'])

    main_df['Buy Day End Quantity'] = main_df['Closing BUY Qty (Cumul)']
    main_df['Sell Day End Quantity'] = main_df['Closing SELL Qty (Cumul)']



    main_df['Open Stock'] =  abs(main_df['Opening BUY Qty (Cumul)'] - main_df['Opening SELL Qty (Cumul)'])
    main_df['Open Stock Avg Price'] = 0
    if transaction_type == 'BUY':
        main_df['Open Stock Avg Price'] = main_df['Opening BUY AVG Price']
    elif transaction_type == 'SELL':
        main_df['Open Stock Avg Price'] = main_df['Opening SELL AVG Price']
    main_df['Open Stock Cost'] = main_df['Open Stock'] * main_df['Open Stock Avg Price']
    main_df['Buy Value'] = main_df['Closing BUY AVG Price'] * main_df['Closing BUY Qty (Cumul)']
    main_df['Sell Value'] = main_df['Closing SELL AVG Price'] * main_df['Closing SELL Qty (Cumul)']
    main_df['Close Qty'] =  main_df['Closing BUY Qty (Cumul)'] - main_df['Closing SELL Qty (Cumul)'] + main_df['Open Stock']
    main_df['Close Stk Cost'] = 0
    if transaction_type == 'BUY':
        main_df['Close Stk Cost'] = np.where(main_df['Close Qty'] > 0 , main_df['Open Stock Cost']+(main_df['Close Qty'] - main_df['Open Stock'])*main_df['Closing BUY AVG Price'],main_df['Close Stk Cost'])
    elif transaction_type == 'SELL':
        main_df['Close Stk Cost'] = np.where(main_df['Close Qty'] < 0 , main_df['Open Stock Cost']+(main_df['Close Qty'] - main_df['Open Stock'])*main_df['Closing SELL AVG Price'],main_df['Close Stk Cost'])


    main_df['Close Stk Avg Cost'] = main_df['Close Stk Cost'] / main_df['Close Qty']
    main_df['Traded Quantity'] = main_df['Close Qty'] - main_df['Open Stock']
    main_df['Profit Today'] =  main_df['Close Stk Cost'] + main_df['Sell Value'] - main_df['Buy Value'] - main_df['Open Stock Cost']

    main_df['Buy Quantity'] = abs(main_df['Buy Day End Quantity'] -  main_df['Opening BUY Qty (Cumul)'])
    main_df['Sell Quantity'] = abs(main_df['Sell Day End Quantity'] - main_df['Opening SELL Qty (Cumul)'])
    main_df['Traded Quantity'] = main_df.apply(lambda x: min(x['Buy Quantity']+x['Open Stock'],x['Sell Quantity']),axis=1)
    main_df['Profit Till Date'] = main_df['Sell Value'] + main_df['Close Stk Cost'] - main_df['Buy Value'] - main_df['Open Stock Cost']

    main_df['Profit Today'] = main_df['Profit Till Date']
    main_df['Profit Today'] = main_df['Profit Today'] -  main_df['Profit Today'].shift(1)
    main_df.loc[0,'Profit Today'] = main_df.loc[0,'Profit Till Date']

    main_df['Day Close'] = main_df['LTP']
    main_df['M2M'] = (main_df['Day Close'] - main_df['Close Stk Avg Cost'])*main_df['Close Qty']
    main_df['Drawdown at EOD'] = main_df['M2M'] +  main_df['Profit Till Date']
    main_df['Investment'] = main_df['Profit Today'] + main_df['Close Stk Cost']
    main_df['OI Closing for the Day'] = main_df['OI']
    main_df['My OI as a % of OI'] = (main_df['Close Qty']*100/main_df['OI']).round(2)
    main_df['Turnover Exchange in Lakhs'] = (main_df['Turnover']/1e6).round(2)
    main_df['My Turnover as Exchange Percentage'] = (main_df['Buy Value']+main_df['Sell Value'])*100/main_df['Turnover']
    main_df['My Turnover as Exchange Percentage'] = main_df['My Turnover as Exchange Percentage'].round(2)
    main_df = main_df[['Date',
       'Closing BUY AVG Price', 'Buy Quantity', 'Closing SELL AVG Price', 'Sell Quantity',
        'OI',
        'Open Stock', 'Open Stock Avg Price',
       'Open Stock Cost', 'Buy Value', 'Sell Value', 'Close Qty',
       'Close Stk Cost', 'Close Stk Avg Cost', 'Traded Quantity',
       'Profit Today', 'Profit Till Date', 'Day Close', 'M2M',
       'Drawdown at EOD', 'Investment', 'OI Closing for the Day',
       'My OI as a % of OI','Turnover Exchange in Lakhs','My Turnover as Exchange Percentage']]
    return main_df



def summary(analysis):
    summary_df = pd.DataFrame()
    temp_dict = dict()
    temp_dict['Start Date'] = analysis['Date'].head(1).values[0]
    temp_dict['End Date'] = analysis['Date'].tail(1).values[0]
    temp_dict['Max Open Qty'] = analysis['Close Qty'].max()
    temp_dict['Max Value of Positions'] = analysis['Investment'].max()
    temp_dict['Max Drawdown'] = analysis['Drawdown at EOD'].min()
    temp_dict['PNL Realized'] = analysis['Profit Till Date'].tail(1).values[0]
    temp_dict['M2M'] = 0 if np.isnan(analysis['M2M'].tail(1).values[0]) else analysis['M2M'].tail(1).values[0]
    temp_dict['Total PNL'] = temp_dict['PNL Realized'] + temp_dict['M2M']
    temp_dict['Charges'] = 0.0001756 * temp_dict['Total PNL']
    temp_dict['Average Turnover'] = analysis['My Turnover as Exchange Percentage'].mean()
    temp_dict['Max Turnover'] = analysis['My Turnover as Exchange Percentage'].max()
    temp_dict['Avg OI'] = analysis['My OI as a % of OI'].mean()
    temp_dict['Max OI'] = analysis['My OI as a % of OI'].max()
    summary_df = summary_df.append(temp_dict,ignore_index=True)
    summary_df = summary_df[['Start Date','End Date','Max Open Qty','Max Value of Positions','Max Drawdown', 'Average Turnover','Max Turnover', 'Avg OI','Max OI',   'Charges',  'M2M',
        'PNL Realized',  'Total PNL']]
    return summary_df

def beautify(backtest):
    remove_cols = []
    for x in list(backtest.columns):
        if backtest[x].dtypes == 'float64':
            backtest[x] = backtest[x].round(2)

        if backtest[x].dtypes == '<M8[ns]':
            print("Changing "+x)
            backtest[x] = backtest[x].dt.strftime('%d/%m/%y %H:%M:%S')

    return backtest


def run_this(ticker_name='MANAA', start_price=105, transaction_type='BUY',
                            entry_diff=.5, exit_diff=1, max_open_lot=20, max_loss=None,
                            lot_size=6000, lot_each_step=1,file_name = 'TEMP',start_date='1/2/2020',end_date='20/2/2020',future_month=None):


    df = get_data(ticker_name=ticker_name,start_date=start_date,end_date=end_date,future_month=future_month)
    trades_df, ledger = do_backtest(df = df,ticker_name=ticker_name, start_price=start_price, transaction_type=transaction_type,
                                    entry_diff=entry_diff, exit_diff=exit_diff, max_open_lot=max_open_lot, max_loss=max_loss,
                                    lot_size=lot_size, lot_each_step=lot_each_step)

    writer = pd.ExcelWriter(file_name+'.xlsx', engine='xlsxwriter')
    analysis = generate_report(ledger,'BUY')
    summary_df = summary(analysis).T
    summary_df.to_excel(writer,sheet_name='Summary')
    trades_df.to_excel(writer,sheet_name='Trade_Sheet')
    ledger.to_excel(writer,sheet_name='Ledger_Main')
    analysis.to_excel(writer, sheet_name='Analysis_Sheet')
    ledger = ledger[(ledger['Buy Qty (Ank)'] != 0) | (ledger['Sell Qty (Ank)'] != 0)]
    ledger.to_excel(writer, sheet_name='Ledger_Filtered')
    writer.close()
    return

if __name__ == '__main__':
    run_this(ticker_name='RBLBANK', start_price=110, transaction_type='BUY',
                            entry_diff=2, exit_diff=2, max_open_lot=30, max_loss=None,
                            lot_size=1500, lot_each_step=1,file_name = 'RBL040420TO200520',start_date='03/04/2020',end_date='20/5/2020',future_month='I')
