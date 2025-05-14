import torch
from Loader import Loader
'''
    포지션 그래프 시작부터 Close Long with Loss 표시되는 버그 수정
'''
if __name__ == "__main__":
    env_path = 'data/preprocess/BTCUSDT/BTCUSDT-30m-HEIKIN_ASHI_200EMA.csv'
    env_path_test = 'data/preprocess/BTCUSDT/BTCUSDT-30m-HEIKIN_ASHI_200EMA_test2.csv'
    
    while True: 
        print('학습 방법을 선택해주세요.')
        print('(1) 학습')
        print('(2) 재개')
        print('(3) 추가 학습')
        print('(4) 테스트')
        print('(5) 디바이스 확인')
        print('(6) 종료')
        
        choice = int(input('선택: '))
        
        if choice == 1:
            loader = Loader(env_path, env_path_test)
            loader.train()
            loader.test()
            break

        elif choice == 2:
            loader = Loader(env_path, env_path_test, further=False)
            loader.train()
            loader.test()
            break

        elif choice == 3:
            loader = Loader(env_path, env_path_test, further=True)
            loader.train()
            loader.test()
            break
        
        elif choice == 4:
            loader = Loader(env_path, env_path_test, further=True)
            loader.test()
            break

        elif choice == 5:
            print("\n현재 사용 가능한 Device : ", "cuda\n" if torch.cuda.is_available() else "cpu\n")

        elif choice == 6:
            print("\n프로그램을 종료합니다.\n")
            exit()

