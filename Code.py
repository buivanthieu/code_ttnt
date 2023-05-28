import math
import os

import codecs


# Hàm words(filename) nhận vào một tên file và trả về một danh sách các từ trong file đó.
def words(filename):
    # Hàm đọc file sử dụng codecs.open() để mở file với mã hóa UTF-8 và bỏ qua các lỗi.
	# Sau đó, các dòng trong file được đọc và chia thành các từ, sau đó được chuyển thành chữ thường và thêm vào danh sách kết quả.

    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as infile:
        lines = infile.readlines()
        
    # Cuối cùng, danh sách các từ được trả về.
    return [word.strip().lower() for line in lines for word in line.split()]


# Hàm lexicon(k) tạo ra hai phân phối: một phân phối cho các từ trong thư mục "spamtraining" và một phân phối cho các từ trong thư mục "hamtraining".

def lexicon(k):
    # Đường dẫn đến các thư mục huấn luyện "spamtraining" và "hamtraining" được trích xuất.
    spam_training_directory = os.getcwd() + '/emails/spamtraining'
    ham_training_directory = os.getcwd() + '/emails/hamtraining'

    # tạo ra các danh sách từ và đếm số lần xuất hiện của từng từ trong các spam email
    spam_distribution = {}
    files = os.listdir(spam_training_directory)
    for file in files:
        list_of_words = words(spam_training_directory + '/' + file)
        for word in list_of_words:
            if word in spam_distribution:
                spam_distribution[word] += 1
            else:
                spam_distribution[word] = 1

    # tạo ra các danh sách từ và đếm số lần xuất hiện của từng từ trong các ham email
    ham_distribution = {}
    files = os.listdir(ham_training_directory)
    for file in files:
        list_of_words = words(ham_training_directory + '/' + file)
        for word in list_of_words:
            if word in ham_distribution:
                ham_distribution[word] += 1
            else:
                ham_distribution[word] = 1

   
    hamkeys = list(ham_distribution.keys())
    spamkeys = list(spam_distribution.keys())
    
	# Các từ có số lần xuất hiện dưới k được loại bỏ khỏi phân phối.
	# Hai phân phối ham và spam được trả về.
    for key in spamkeys:
        if spam_distribution[key] < k:
            del spam_distribution[key]

    for key in hamkeys:
        if ham_distribution[key] < k:
            del ham_distribution[key]

    return ham_distribution, spam_distribution



# Hàm probability tính xác suất P(w = word | category) với công thức smoothing Laplacian và tham số m.

def probability(word, category, ham_distribution, spam_distribution, m):
    	
	# Đầu tiên, phân phối tương ứng được chọn (ham hoặc spam) dựa trên tham số "category".
	# Biến V lưu trữ số lượng từ duy nhất trong phân phối.
	# Biến keys lưu trữ danh sách các từ trong phân phối.
	distribution = ham_distribution if category == 'ham' else spam_distribution

	V = len(distribution)

	keys = distribution.keys()
        
	# Số phần tử trong tử số và mẫu số được tính toán và trả về xác suất.

	numerator = (distribution[word] + m if word in keys else m)
	denominator = sum([distribution[key] for key in keys]) + m*V

	return numerator / float(denominator)


# Hàm classify_email xác định xem một email có được phân loại là "ham" hay "spam" dựa trên các xác suất từ các phân phối và tham số m.


def classify_email(email, ham_distribution, spam_distribution, m):
    	
    # Các từ trong email được lấy ra bằng cách sử dụng hàm words(email).
	email_words = words(email)

	ham_probability  = 0
	spam_probability = 0
	# Xác suất cho cả hai loại (ham và spam) được tính bằng cách thêm các log xác suất của các từ trong email.

	for word in email_words:
		ham_probability  += math.log(probability(word, 'ham', ham_distribution, spam_distribution, m))
		spam_probability += math.log(probability(word, 'spam', ham_distribution, spam_distribution, m))

	#hàm trả về "ham" nếu xác suất cho "ham" lớn hơn xác suất cho "spam", ngược lại trả về "spam".
	return 'ham' if ham_probability > spam_probability else 'spam'



# Hàm test_filter kiểm tra hiệu suất của bộ lọc bằng cách đánh giá các email trong tập kiểm tra.


def test_filter(hamtesting, spamtesting, k, m):
    	
	# phân phối ham và spam được tạo bằng cách sử dụng hàm lexicon(k).
	ham_distribution, spam_distribution = lexicon(k)

	# Các biến và danh sách được khởi tạo để theo dõi kết quả phân loại.
	spam_as_ham = []
	ham_as_spam = []

	ham_hit   = 0
	ham_total = 0
	ham_testing_files = os.listdir(hamtesting)
        
		# Vòng lặp đi qua các file trong thư mục kiểm tra "hamtesting" và tăng số lượng đúng (ham_hit) hoặc thêm file vào danh sách ham_as_spam nếu phân loại sai.
	for file in ham_testing_files:
		if classify_email(hamtesting + '/' + file, ham_distribution, spam_distribution, m) == 'ham':
			ham_hit += 1
		else:
			ham_as_spam.append(file)
		ham_total += 1

	spam_hit   = 0
	spam_total = 0
	spam_testing_files = os.listdir(spamtesting)
        # Tương tự, vòng lặp thứ hai đối với các file trong thư mục kiểm tra "spamtesting".

	for file in spam_testing_files:
		if classify_email(spamtesting + '/' + file, ham_distribution, spam_distribution, m) == 'spam':
			spam_hit += 1
		else:
			spam_as_ham.append(file)
		spam_total += 1

	ham_hit_ratio  = ham_hit / float(ham_total)
	spam_hit_ratio = spam_hit / float(spam_total)
        

	# Tỉ lệ đúng cho cả hai loại (ham và spam) được tính và trả về.
	return ham_hit_ratio, spam_hit_ratio, ham_total, spam_total, ham_as_spam, spam_as_ham


#Hàm kiểm tra hiệu suất và phân loại thư phần testing
def recognize_email_on_folder():
	spamtesting = os.getcwd() + '/emails/spamtesting'
	hamtesting  = os.getcwd() + '/emails/hamtesting'

	ham_hit_ratio, spam_hit_ratio, ham_total, spam_total, ham_as_spam, spam_as_ham = test_filter(hamtesting, spamtesting, k=5, m=1)

	print ()
	print ("Tỉ lệ thư đúng chính xác: ", ham_hit_ratio * 100)
	print ("Tỉ lệ thư rác chính xác:    ", spam_hit_ratio * 100)
	print ("Tỉ lệ nhận biết đúng: ", (ham_hit_ratio*ham_total + spam_hit_ratio*spam_total) / (ham_total + spam_total) * 100)

	print ("\nCác thư đúng nhưng gắn nhãn thư rác:")
	for file in ham_as_spam:
		print ("\t"+file)

	print ("\nCác thư rác nhưng gắn nhãn thư đúng:")
	for file in spam_as_ham:
		print ("\t"+file)
	print()
        

#Hàm kiểm tra thư nhập input là hợp lệ hay không?		
def classify_email_from_input(ham_distribution, spam_distribution, m):
    email = input("Nhập nội dung email: ")
    
    ham_probability = 0
    spam_probability = 0

    email_words = email.split()  # Chia nội dung email thành danh sách các từ

    for word in email_words:
        ham_probability += math.log(probability(word, 'ham', ham_distribution, spam_distribution, m))
        spam_probability += math.log(probability(word, 'spam', ham_distribution, spam_distribution, m))

    classification = 'Email hợp lệ' if ham_probability > spam_probability else 'Email là Spam'
    print("Phân loại: ", classification)


ham_distribution, spam_distribution = lexicon(5)

def main():
    ham_distribution, spam_distribution = lexicon(k=5)
    
    print("1. Phân loại email từ input")
    print("2. Kiểm tra hiệu suất của bộ lọc trên thư mục")
    choice = input("Chọn một tác vụ (1/2): ")
    
    if choice == "1":
        classify_email_from_input(ham_distribution, spam_distribution, m=1)
    elif choice == "2":
        recognize_email_on_folder()
    else:
        print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
