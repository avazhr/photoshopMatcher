from arcfaceEmb import ArcFaceEmb

if __name__ == "main":

    a1 = ArcfaceEmb.get_embedding("../Originals/02463d468.jpg")
    a2 = ArcfaceEmb.get_embedding("../Originals/02463d469.jpg")

    b1 = ArcfaceEmb.get_embedding("../Originals/04201d320.jpg")
    
    print("a1 vs a2: ")
    print(ArcfaceEmb.get_embedding(a1, a2))
    print("a1 vs b1: ")
    print(ArcfaceEmb.get_embedding(a1, b1))
