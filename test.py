from blimp import Blimp



def run():
    env = Blimp("sano.xml", render_mode="human")

    a = [0,0,0,0,0,0]
    while True:
        ob, reward, terminated,_ = env.step(a)
        print(ob)


if __name__ == '__main__':
    run()


