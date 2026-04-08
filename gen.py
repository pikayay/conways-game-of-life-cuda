import matplotlib.pyplot as plt

binary = binary.reshape((1024, 1024))
plt.imshow(binary, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.savefig("output.png", bbox_inches='tight', pad_inches=0)