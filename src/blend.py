import pandas as pd

def blend():
    sub_nn = pd.read_csv("spaceship-titanic/submission_nn.csv")
    sub_base = pd.read_csv("spaceship-titanic/submission_baseline.csv")
    
    blend_pred = (0.6 * sub_nn["Transported"].astype(int) + 
                  0.4 * sub_base["Transported"].astype(int))
    
    final_preds = (blend_pred >= 0.5)
    
    submission = pd.DataFrame({
        "PassengerId": sub_nn["PassengerId"],
        "Transported": final_preds
    })
    
    submission.to_csv("spaceship-titanic/submission_blend.csv", index=False)
    print("Wrote blended submission to spaceship-titanic/submission_blend.csv")

if __name__ == "__main__":
    blend()