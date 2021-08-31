
while True:
    try:
        n = input().split()
    except Exception:
        break
    print(abs(int(n[0]) - int(n[1])))