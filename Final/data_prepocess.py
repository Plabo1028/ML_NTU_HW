import numpy as np
import cPickle
import time
import datetime
import csv


def student_enrollment(
    enroll_train_path='Data/enrollment_train.csv',
    enroll_test_path='Data/enrollment_test.csv',
    ):
    student_enrollment = {}

    f = open(enroll_train_path)
    data = f.read()
    data = data.split('\n')
    for line in data[1:-1]:
        enrollment_id,username,course_id = line.split(',')
        if username not in student_enrollment:
            student_enrollment[username] = [enrollment_id]
        else:
            student_enrollment[username].append(enrollment_id)
            # raw_input(student_enrollment[username])
    f.close()

    f = open(enroll_test_path)
    data = f.read()
    data = data.split('\n')
    for line in data[1:-1]:
        enrollment_id,username,course_id = line.split(',')
        if username not in student_enrollment:
            student_enrollment[username] = [enrollment_id]
        else:
            student_enrollment[username].append(enrollment_id)
            # raw_input(student_enrollment[username])
    f.close()

    w = open("student_enrollment.cPickle",'w')
    cPickle.dump(student_enrollment,w)
    w.close()

def student2enrollid_time():

    '''initial'''
    student2enrollid_time = {}
    enrollid_second = cPickle.load(open('enrollid_second.cPickle'))
    student_enrollment = cPickle.load(open('student_enrollment.cPickle'))

    for key,values in student_enrollment.items():
        # print key,values
        max_time = 0
        for enrollid in values:
            secods = enrollid_second.get(enrollid)
            print secods
            if secods > max_time:
                max_time = secods

        for enrollid in values:
            student2enrollid_time[enrollid] = max_time
        # raw_input(student2enrollid_time)

    w = open("student2enrollid_time.cPickle",'w')
    cPickle.dump(student2enrollid_time,w)
    w.close()

    w = open('student2enrollid_time.txt','w')
    w.write('enrollment_id,student2enrollid_time\n')
    for key in sorted(student2enrollid_time.iterkeys()):
        w.write(str(key))
        w.write(',')
        w.write(str(student2enrollid_time[key]))
        w.write('\n')
    w.close()

    # return student2enrollid_time

def enrollment_time(
    enroll_train_path='Data/enrollment_train.csv',
    enroll_test_path='Data/enrollment_test.csv',
    log_train_path='Data/log_train.csv',
    log_test_path='Data/log_test.csv',
    ):

    def dis_seconds(distance):
        # print 'distance',distance
        time = str(distance).split(' ')
        if len(time) != 1:
            day = time[0]
            # print day
            dis_seconds = distance.seconds + (int(day) * 86400)
        else:
            dis_seconds = distance.seconds
        # print 'dis_seconds',dis_seconds
        return int(dis_seconds)

    '''initial'''
    enrollid_time = {}
    enrollid_second = {}

    f = open(log_train_path)
    data = f.read()
    data = data.split('\n')
    last_time_enrollment = 0
    for line in data[1:-1]:
        '''tmp = enrollment_id,time'''
        tmp = line.split(',')[:2]

        if tmp[0] not in enrollid_time:
            now_time = datetime.datetime.strptime(tmp[1], '%Y-%m-%dT%H:%M:%S')
            enrollid_time[tmp[0]] = now_time
            '''if in the end ds = 0 means the enrollment attend one time so he drop out'''
            distance = datetime.datetime(2015,12,28)-datetime.datetime(2015,12,28)
            ds = dis_seconds(distance)
            enrollid_second[tmp[0]] = ds
            '''handle ds = 0'''
            try :
                if enrollid_second[last_time_enrollment] == 0:
                    enrollid_second[last_time_enrollment] = int(30*24*3600)
            except:
                print 'Except !!!!!'
                # print tmp
                # raw_input(last_time_enrollment)

        else:
            last_time = enrollid_time.get(tmp[0])
            now_time = datetime.datetime.strptime(tmp[1], '%Y-%m-%dT%H:%M:%S')
            enrollid_time[tmp[0]] = now_time
            # print enrollid_time[tmp[0]]
            '''distance from now_time to last_time'''
            # print 'now_time-last_time',(now_time-last_time)
            if distance < (now_time-last_time):
                distance = (now_time-last_time)
                # print 'now_time',now_time
                # print 'last_time',last_time
                ds = dis_seconds(distance)
                enrollid_second[tmp[0]] = ds
                # raw_input(enrollid_second)
        last_time_enrollment = tmp[0]
    f.close()

    f = open(log_test_path)
    data = f.read()
    data = data.split('\n')
    for line in data[1:-1]:
        '''tmp = enrollment_id,time'''
        tmp = line.split(',')[:2]

        if tmp[0] not in enrollid_time:
            now_time = datetime.datetime.strptime(tmp[1], '%Y-%m-%dT%H:%M:%S')
            enrollid_time[tmp[0]] = now_time
            '''if in the end ds = 0 means the enrollment attend one time so he drop out'''
            distance = datetime.datetime(2015,12,28)-datetime.datetime(2015,12,28)
            ds = dis_seconds(distance)
            enrollid_second[tmp[0]] = ds
            '''handle ds = 0'''
            try :
                if enrollid_second[last_time_enrollment] == 0:
                    enrollid_second[last_time_enrollment] = int(30*24*3600)
            except:
                print 'Except !!!!!'
                # print tmp
                # raw_input(last_time_enrollment)
        else:
            last_time = enrollid_time.get(tmp[0])
            now_time = datetime.datetime.strptime(tmp[1], '%Y-%m-%dT%H:%M:%S')
            enrollid_time[tmp[0]] = now_time
            # print enrollid_time[tmp[0]]
            '''distance from now_time to last_time'''
            # print 'now_time-last_time',(now_time-last_time)
            if distance < (now_time-last_time):
                distance = (now_time-last_time)
                # print 'now_time',now_time
                # print 'last_time',last_time
                ds = dis_seconds(distance)
                enrollid_second[tmp[0]] = ds
                # raw_input(enrollid_second)
        last_time_enrollment = tmp[0]
    f.close()

    w = open("enrollid_second.cPickle",'w')
    cPickle.dump(enrollid_second,w)
    w.close()

    w = open('enrollid_second.txt','w')
    w.write('enrollment_id,enrollid_second\n')
    for key in sorted(enrollid_second.iterkeys()):
        w.write(str(key))
        w.write(',')
        w.write(str(enrollid_second[key]))
        w.write('\n')
    w.close()

    # return enrollid_second

def enrollment_rate(
    enroll_train_path='Data/enrollment_train.csv',
    enroll_test_path='Data/enrollment_test.csv',
    log_train_path='Data/log_train.csv',
    log_test_path='Data/log_test.csv',
    ):

    '''initial'''
    student_enrollment = cPickle.load(open('student_enrollment.cPickle'))
    enrollid2number = {}
    enroll_rate = {}

    f = open(log_train_path)
    data = f.read()
    data = data.split('\n')
    for line in data[1:-1]:
        # print line.split(',')
        enrollment_id = line.split(',')[0]
        if enrollment_id not in enrollid2number:
            enrollid2number[enrollment_id] = 1
        else:
            enrollid2number[enrollment_id] += 1
        # raw_input(enrollid2number)
    f.close()

    f = open(log_test_path)
    data = f.read()
    data = data.split('\n')
    for line in data[1:-1]:
        # print line.split(',')
        enrollment_id = line.split(',')[0]
        if enrollment_id not in enrollid2number:
            enrollid2number[enrollment_id] = 1
        else:
            enrollid2number[enrollment_id] += 1
        # raw_input(enrollid2number)
    f.close()

    for key,values in student_enrollment.items():
        # print 'key',key
        # print 'values',values
        sum_log = 0
        for enrollid in values:
            # print enrollid2number[enrollid]
            if enrollid not in enroll_rate:
                enroll_rate[enrollid] = int(enrollid2number[enrollid])
            else:
                raw_input('double enrollid')
                print 'key',key
                print 'values',values
                # enroll_rate[enrollid].append(enrollid2number[enrollid])
            sum_log = sum_log + enrollid2number[enrollid]
            # raw_input(enroll_rate)
        for enrollid in values:
            enroll_rate[enrollid] /= float(sum_log)
            # raw_input(enroll_rate)

    w = open('enrollment_rate.txt','w')
    w.write('enrollment_id,enrollment_rate\n')
    for key in sorted(enroll_rate.iterkeys()):
        w.write(str(key))
        w.write(',')
        w.write(str(enroll_rate[key]))
        w.write('\n')
    w.close()

    w = open("enrollment_rate.cPickle",'w')
    cPickle.dump(enroll_rate,w)
    w.close()

    # return enroll_rate

if __name__ == '__main__':
    # student_enrollment()

    # enrollment_rate()
    # print 'Done enroll_rate!'

    # enrollment_time()
    # print 'Done enrollment_time!'

    # student2enrollid_time()
    # print 'Done student2enrollid_time!'

    '''wirte sample_train_x.csv and sample_test_x.csv'''
    sample_train_x = open('Data/sample_train_x.csv')
    sample_test_x = open('Data/sample_test_x.csv')
    enrollid_second = cPickle.load(open('enrollid_second.cPickle'))
    enrollment_rate = cPickle.load(open('enrollment_rate.cPickle'))
    student2enrollid_time = cPickle.load(open('student2enrollid_time.cPickle'))

    with open('Data/sample_train_x.csv','r') as csvinput:
        with open('sample_train_x20.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            row = next(reader)
            row.append('enrollment_rate,enrollid_second,student2enrollid_time')
            all.append(row)
            for row in reader:
                row.append(enrollment_rate[row[0]])
                row.append(enrollid_second[row[0]])
                row.append(student2enrollid_time[row[0]])
                all.append(row)
            writer.writerows(all)

    with open('Data/sample_test_x.csv','r') as csvinput:
        with open('sample_test_x20.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            row = next(reader)
            row.append('enrollment_rate,enrollid_second,student2enrollid_time')
            all.append(row)
            for row in reader:
                row.append(enrollment_rate[row[0]])
                row.append(enrollid_second[row[0]])
                row.append(student2enrollid_time[row[0]])
                all.append(row)
            writer.writerows(all)




