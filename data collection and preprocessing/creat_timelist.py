import datetime


start_date = datetime.datetime(2022, 12, 31, 0, 0, 0)
end_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
def creat_timelist(start_date, end_date, step =3):

    date_list = [start_date + datetime.timedelta(hours=step*x) for x in range(int((end_date-start_date).total_seconds()/3600/3))]

    date_strings = [d.strftime('%Y-%m-%dT%H:%M:%SZ') for d in date_list]

    return date_strings
