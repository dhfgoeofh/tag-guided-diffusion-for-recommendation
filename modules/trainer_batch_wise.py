import torch
import torch.nn as nn
import torch.optim as optim
import time

# 배치 내에서 샘플링 수만큼 한번에 학습 (일반화 향상)
class Trainer:
    def __init__(self, model, diffusion, device, num_t_samples, args):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.epochs = args.epochs
        self.best_valid_loss = float('inf')
        self.best_epoch = -1
        self.num_t_samples = num_t_samples
        self.model_path = args.model_path
        
        # optimizer
        if args.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8, weight_decay=args.wd)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optimizer == 'Momentum':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.wd)

        self.optimizer = optimizer

        self.start_time = None
        self.end_time = None

    def train(self, train_loader, valid_loader):
        self.start_time = time.localtime(time.time())
        print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for item_batch, tag_batch in train_loader:
                item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()
                loss = 0
                for _ in range(self.num_t_samples):  # 샘플링 수만큼 반복하여 로스를 누적
                    loss += self.diffusion(item_batch, classes=tag_batch)

                loss = loss / self.num_t_samples  # 평균 손실 계산

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            if epoch % 10 == 0:
                avg_valid_loss = self.validate(valid_loader)

                if avg_valid_loss < self.best_valid_loss:
                    self.best_valid_loss = avg_valid_loss
                    self.best_epoch = epoch + 1

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")
        print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}")

    def validate(self, valid_loader):
        self.model.eval()
        total_valid_loss = 0

        with torch.no_grad():
            for item_batch, tag_batch in valid_loader:
                item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()
                
                # Sample generation using diffusion's sample method
                generated_samples = self.diffusion.sample(classes=tag_batch)
                
                # Calculate loss based on the generated samples and ground truth
                loss = self.calculate_loss(generated_samples, item_batch)
                total_valid_loss += loss.item()

        return total_valid_loss / len(valid_loader)

    def test(self, test_loader):
        self.model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for item_batch, tag_batch in test_loader:
                item_batch, tag_batch = item_batch.cuda(), tag_batch.cuda()
                
                # Sample generation using diffusion's sample method
                generated_samples = self.diffusion.sample(classes=tag_batch)
                
                # Calculate loss based on the generated samples and ground truth
                loss = self.calculate_loss(generated_samples, item_batch)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss}")
        print(f"Best Epoch: {self.best_epoch}, Best Validation Loss: {self.best_valid_loss}")
        self.end_time = time.localtime(time.time())
        
        # 시간을 초 단위로 변환 후 차이를 계산
        start_seconds = time.mktime(self.start_time)
        end_seconds = time.mktime(self.end_time)
        runtime = end_seconds - start_seconds
        
        print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', self.end_time), 'runtime:', time.strftime('%H:%M:%S', time.gmtime(runtime)))

    def calculate_loss(self, generated_samples, ground_truth):
        """
        Helper function to calculate loss between generated samples and ground truth
        """
        return nn.MSELoss()(generated_samples, ground_truth)
    
    def save(self):
        data = {
            'epochs': self.epochs,
            'model': self.model.get_state_dict(self.model)
        }

        torch.save(data, str(self.model_path / f'model-{self.epochs}_epoch.pt'))