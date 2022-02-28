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


public class userLogInActivity extends AppCompatActivity {

    private static final String TAG = "SignInActivity";
    public FirebaseAuth mAuth;
    Button btnLogIn;
    TextView txtBtnSignUp;
    EditText email,pass;

    @SuppressLint("WrongViewCast")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_log_in);

        btnLogIn= findViewById(R.id.btnLogIn);
        btnLogIn.setTextColor(Color.parseColor("#FFFFFF"));
        btnLogIn.setBackgroundColor(Color.parseColor("#0000FF"));
        email=findViewById(R.id.editTxtEmailLogIn);
        pass=findViewById(R.id.editTxtPasswordLogIn);
        txtBtnSignUp = findViewById(R.id.txtBtnSignUp);
        txtBtnSignUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent2 = new Intent(userLogInActivity.this , userSignUpActivity.class);
                startActivity(intent2);
            }
        });

        mAuth = FirebaseAuth.getInstance();

        btnLogIn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (email.getText().toString().contentEquals("")) {


                    Toast.makeText(getApplicationContext(),"Email cannot be empty",Toast.LENGTH_SHORT).show();


                } else if (pass.getText().toString().contentEquals("")) {

                    Toast.makeText(getApplicationContext(),"Password cannot be empty",Toast.LENGTH_SHORT).show();

                } else {


                    mAuth.signInWithEmailAndPassword(email.getText().toString(), pass.getText().toString())
                            .addOnCompleteListener(userLogInActivity.this, new OnCompleteListener<AuthResult>() {
                                @Override
                                public void onComplete(@NonNull Task<AuthResult> task) {
                                    if (task.isSuccessful()) {
                                        // Sign in success, update UI with the signed-in user's information
                                        Log.d(TAG, "signInWithEmail:success");

                                        FirebaseUser user = mAuth.getCurrentUser();
                                        Intent HomeActivity = new Intent(userLogInActivity.this, MenuActivity.class);
                                        setResult(RESULT_OK, null);
                                        startActivity(HomeActivity);
                                        userLogInActivity.this.finish();

                                    } else {
                                        // If sign in fails, display a message to the user.
                                        Log.w(TAG, "signInWithEmail:failure", task.getException());
                                        Toast.makeText(userLogInActivity.this, "Authentication failed.",
                                                Toast.LENGTH_SHORT).show();
                                        if (task.getException() != null) {
                                            Toast.makeText(getApplicationContext(),"Error  Retry",Toast.LENGTH_SHORT).show();
                                        }

                                    }

                                }
                            });


                }


            }
        });



    }
}
