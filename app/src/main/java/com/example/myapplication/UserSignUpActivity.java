package com.example.myapplication;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;


public class UserSignUpActivity extends AppCompatActivity {


    private static final String TAG = "SignUpActivity";
    public FirebaseAuth mAuth;
    Button btnSignUp;
    TextView txtBtnLogIn;
    EditText signUpEmailTextInput;
    EditText signUpPasswordTextInput;
    TextView errorView;

    @SuppressLint("WrongViewCast")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_sign_up);


        //Connecting with the firebase
        mAuth = FirebaseAuth.getInstance();

        btnSignUp = findViewById(R.id.btnSignUp);
        btnSignUp.setTextColor(Color.parseColor("#FFFFFF"));
        btnSignUp.setBackgroundColor(Color.parseColor("#0000FF"));
        signUpEmailTextInput=findViewById(R.id.editTxtEmail);
        signUpPasswordTextInput=findViewById(R.id.editTxtPassword);
        txtBtnLogIn = findViewById(R.id.txtBtnLogIn);
        txtBtnLogIn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent1 = new Intent(UserSignUpActivity.this , UserLogInActivity.class);
                startActivity(intent1);
            }
        });


        btnSignUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (signUpEmailTextInput.getText().toString().contentEquals("")) {


                    errorView.setText("Email cannot be empty");


                } else if (signUpPasswordTextInput.getText().toString().contentEquals("")) {


                    errorView.setText("Password cannot be empty");


                }


                 else {


                    mAuth.createUserWithEmailAndPassword(signUpEmailTextInput.getText().toString(), signUpPasswordTextInput.getText().toString()).addOnCompleteListener(UserSignUpActivity.this, new OnCompleteListener<AuthResult>() {
                        @Override
                        public void onComplete(@NonNull Task<AuthResult> task) {

                            if (task.isSuccessful()) {
                                // Sign in success, update UI with the signed-in user's information
                                Log.d(TAG, "createUserWithEmail:success");
                                FirebaseUser user = mAuth.getCurrentUser();
                                Intent signInIntent = new Intent(UserSignUpActivity.this, UserLogInActivity.class);
                                UserSignUpActivity.this.finish();
                                startActivity(signInIntent);

                            } else {
                                // If sign in fails, display a message to the user.
                                Log.w(TAG, "createUserWithEmail:failure", task.getException());
                                Toast.makeText(UserSignUpActivity.this, "Authentication failed.",
                                        Toast.LENGTH_SHORT).show();

                                if (task.getException() != null) {
                                     Toast.makeText(getApplicationContext(),"Error",Toast.LENGTH_SHORT).show();

                                }

                            }

                        }
                    });

                }

            }
        });



    }
}
