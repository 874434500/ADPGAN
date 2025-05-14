# 假设每个参数是 32 位浮点数（4 字节）

total_params=4355376


param_size = total_params * 4  # 参数大小（字节）

# 输入数据大小
batch_size = 16
input_shape = (144, 144, 3)
input_size = batch_size * input_shape[0] * input_shape[1] * input_shape[2] * 4  # 输入数据大小（字节）

# 中间结果大小（假设为输入数据大小的 2 倍）
intermediate_size = 2 * input_size  # 中间结果大小（字节）

# 梯度大小（与参数大小相同）
gradient_size = param_size  # 梯度大小（字节）

# 优化器状态大小（假设为参数大小的 2 倍）
optimizer_state_size = 2 * param_size  # 优化器状态大小（字节）

# 总内存需求
total_memory需求 = param_size + input_size + intermediate_size + gradient_size + optimizer_state_size

# 打印内存需求
print(f"模型参数大小: {param_size / (1024 * 1024):.2f} MB")
print(f"输入数据大小: {input_size / (1024 * 1024):.2f} MB")
print(f"中间结果大小: {intermediate_size / (1024 * 1024):.2f} MB")
print(f"梯度大小: {gradient_size / (1024 * 1024):.2f} MB")
print(f"优化器状态大小: {optimizer_state_size / (1024 * 1024):.2f} MB")
print(f"总内存需求: {total_memory需求 / (1024 * 1024):.2f} MB")