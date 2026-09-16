![Bits precision](https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Faccelerating-ai-training-with-tf32-tensor-cores%2F&ved=0CBYQjRxqFwoTCIisk5il85YDFQAAAAAdAAAAABBv&opi=89978449)

### auto casting with the torch.autocase
it takes some parameter and convert that into the defined precision

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16): #to auto cast here to add the dtype=?
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()
```

### ugly numbers
cuda is otpimal for power of 2 and even nbrs and very worst optimized for prime nbrs and odd nbrs , 
so we try to use them very less in our code for optimal training
